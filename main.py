"""
Local Base Obstacle Avoidance Service — TidyBot Backend
Maintains a local 2D costmap from depth images and provides safe velocity
commands using the Dynamic Window Approach (DWA). Pure Python, no ROS.
"""

import base64
import io
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ─── Configuration ────────────────────────────────────────────────
ROBOT_RADIUS = float(os.getenv("ROBOT_RADIUS", 0.18))          # meters
MIN_OBSTACLE_DIST = float(os.getenv("MIN_OBSTACLE_DIST", 0.25))  # meters, e-stop threshold
MAX_RANGE = float(os.getenv("MAX_RANGE", 4.0))                  # meters
COSTMAP_SIZE = int(os.getenv("COSTMAP_SIZE", 200))               # cells per side
COSTMAP_RES = float(os.getenv("COSTMAP_RES", 0.025))            # meters per cell (5m / 200)

# DWA parameters
DWA_MAX_VEL = float(os.getenv("DWA_MAX_VEL", 0.5))       # m/s
DWA_MAX_OMEGA = float(os.getenv("DWA_MAX_OMEGA", 1.5))    # rad/s
DWA_VEL_SAMPLES = int(os.getenv("DWA_VEL_SAMPLES", 15))
DWA_OMEGA_SAMPLES = int(os.getenv("DWA_OMEGA_SAMPLES", 21))
DWA_DT = float(os.getenv("DWA_DT", 0.3))                  # sim timestep
DWA_HORIZON = float(os.getenv("DWA_HORIZON", 1.5))         # seconds

# ─── Global State ─────────────────────────────────────────────────
costmap: np.ndarray = None  # float32 HxW, 0=free, 1=occupied
costmap_lock = None
estop_active = False
last_update_time = 0.0


def init_costmap():
    global costmap
    costmap = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE), dtype=np.float32)


def world_to_grid(x: float, y: float):
    """Convert robot-frame coords (x=forward, y=left) to grid indices. Robot is at center."""
    cx, cy = COSTMAP_SIZE // 2, COSTMAP_SIZE // 2
    gi = cy - int(round(x / COSTMAP_RES))
    gj = cx - int(round(y / COSTMAP_RES))
    return gi, gj


def grid_to_world(gi: int, gj: int):
    cx, cy = COSTMAP_SIZE // 2, COSTMAP_SIZE // 2
    x = (cy - gi) * COSTMAP_RES
    y = (cx - gj) * COSTMAP_RES
    return x, y


# ─── Depth → Costmap ─────────────────────────────────────────────
def depth_to_costmap(depth_img: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """Project depth image into local 2D costmap (robot frame, bird's eye)."""
    global costmap, estop_active, last_update_time

    h, w = depth_img.shape[:2]
    # Decay old costmap
    costmap *= 0.6

    min_dist_found = float('inf')

    # Subsample for speed
    step = max(1, min(h, w) // 80)
    vs = np.arange(0, h, step)
    us = np.arange(0, w, step)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.flatten()
    vv = vv.flatten()

    depths = depth_img[vv, uu].astype(np.float32)

    # Handle mm vs m (if max > 20, assume mm)
    if depths.max() > 20.0:
        depths = depths / 1000.0

    valid = (depths > 0.05) & (depths < MAX_RANGE)
    uu, vv, depths = uu[valid], vv[valid], depths[valid]

    if len(depths) == 0:
        last_update_time = time.time()
        return

    # Back-project to 3D (camera frame: z=forward, x=right, y=down)
    z = depths
    x_cam = (uu.astype(np.float32) - cx) * z / fx
    y_cam = (vv.astype(np.float32) - cy) * z / fy

    # Filter to ground-relevant heights (assume camera ~0.5-1.5m high, obstacles 0.02-1.5m)
    # We keep points that aren't floor (y_cam not too large) and not ceiling
    # y_cam > 0 means below optical center → could be floor or low obstacles
    # Simple: keep all, project to 2D top-down (x_robot=z_cam, y_robot=-x_cam)
    x_robot = z        # forward
    y_robot = -x_cam   # left

    # Find minimum distance
    dists = np.sqrt(x_robot**2 + y_robot**2)
    if len(dists) > 0:
        min_dist_found = float(np.min(dists))

    # Project to grid
    gi_arr = (COSTMAP_SIZE // 2 - (x_robot / COSTMAP_RES)).astype(np.int32)
    gj_arr = (COSTMAP_SIZE // 2 - (y_robot / COSTMAP_RES)).astype(np.int32)

    in_bounds = (gi_arr >= 0) & (gi_arr < COSTMAP_SIZE) & (gj_arr >= 0) & (gj_arr < COSTMAP_SIZE)
    gi_arr, gj_arr = gi_arr[in_bounds], gj_arr[in_bounds]

    # Mark occupied
    np.add.at(costmap, (gi_arr, gj_arr), 0.4)
    np.clip(costmap, 0, 1.0, out=costmap)

    estop_active = min_dist_found < MIN_OBSTACLE_DIST
    last_update_time = time.time()


# ─── DWA Planner ──────────────────────────────────────────────────
def check_collision(x, y, theta, costmap_arr):
    """Check if position (robot frame) collides with costmap obstacles."""
    gi, gj = world_to_grid(x, y)
    r_cells = max(1, int(math.ceil(ROBOT_RADIUS / COSTMAP_RES)))
    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            ni, nj = gi + di, gj + dj
            if 0 <= ni < COSTMAP_SIZE and 0 <= nj < COSTMAP_SIZE:
                if costmap_arr[ni, nj] > 0.5:
                    return True
    return False


def dwa_plan(target_vx: float, target_vy: float, target_omega: float,
             goal_x: Optional[float] = None, goal_y: Optional[float] = None):
    """
    Dynamic Window Approach: sample velocities, simulate trajectories,
    score by goal heading + velocity + obstacle clearance.
    Returns (safe_vx, safe_omega, trajectories_info).
    """
    global costmap

    if costmap is None:
        return target_vx, target_omega, {}

    best_score = -1e9
    best_v = 0.0
    best_w = 0.0

    v_min, v_max = -0.1, DWA_MAX_VEL
    w_min, w_max = -DWA_MAX_OMEGA, DWA_MAX_OMEGA

    dt = DWA_DT
    n_steps = max(1, int(DWA_HORIZON / dt))

    for vi in np.linspace(v_min, v_max, DWA_VEL_SAMPLES):
        for wi in np.linspace(w_min, w_max, DWA_OMEGA_SAMPLES):
            # Simulate trajectory
            x, y, theta = 0.0, 0.0, 0.0
            collision = False
            min_obs_dist = float('inf')

            for _ in range(n_steps):
                x += vi * math.cos(theta) * dt
                y += vi * math.sin(theta) * dt
                theta += wi * dt

                if check_collision(x, y, theta, costmap):
                    collision = True
                    break

                # Distance to nearest obstacle
                gi, gj = world_to_grid(x, y)
                r_check = int(1.0 / COSTMAP_RES)
                for di in range(-r_check, r_check + 1, 2):
                    for dj in range(-r_check, r_check + 1, 2):
                        ni, nj = gi + di, gj + dj
                        if 0 <= ni < COSTMAP_SIZE and 0 <= nj < COSTMAP_SIZE:
                            if costmap[ni, nj] > 0.5:
                                d = math.sqrt((di * COSTMAP_RES)**2 + (dj * COSTMAP_RES)**2)
                                min_obs_dist = min(min_obs_dist, d)

            if collision:
                continue

            # Scoring
            # 1) Velocity: prefer matching target velocity
            vel_score = 1.0 - abs(vi - target_vx) / max(DWA_MAX_VEL, 0.01)

            # 2) Heading: prefer matching target omega or heading to goal
            if goal_x is not None and goal_y is not None:
                goal_angle = math.atan2(goal_y, goal_x)
                heading_score = 1.0 - abs(theta - goal_angle) / math.pi
            else:
                heading_score = 1.0 - abs(wi - target_omega) / max(DWA_MAX_OMEGA, 0.01)

            # 3) Clearance
            clearance_score = min(min_obs_dist, 1.0)

            score = 2.0 * vel_score + 3.0 * heading_score + 1.5 * clearance_score

            if score > best_score:
                best_score = score
                best_v = vi
                best_w = wi

    return float(best_v), float(best_w), {"score": float(best_score)}


# ─── FastAPI App ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_costmap()
    print(f"Local obstacle avoidance ready. Costmap {COSTMAP_SIZE}x{COSTMAP_SIZE}, "
          f"res={COSTMAP_RES}m, robot_radius={ROBOT_RADIUS}m, max_range={MAX_RANGE}m")
    yield


app = FastAPI(
    title="Local Base Obstacle Avoidance Service",
    description="Real-time local obstacle avoidance using depth images and DWA.",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────
class UpdateDepthRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded depth image (16-bit PNG or float)")
    fx: float = Field(..., description="Camera focal length x (pixels)")
    fy: float = Field(..., description="Camera focal length y (pixels)")
    cx: float = Field(..., description="Camera principal point x (pixels)")
    cy: float = Field(..., description="Camera principal point y (pixels)")


class SafeVelocityRequest(BaseModel):
    target_vx: float = Field(0.0, description="Desired forward velocity (m/s)")
    target_omega: float = Field(0.0, description="Desired angular velocity (rad/s)")
    goal_x: Optional[float] = Field(None, description="Optional goal x in robot frame (m)")
    goal_y: Optional[float] = Field(None, description="Optional goal y in robot frame (m)")


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "local-obstacle-avoidance",
        "costmap_size": COSTMAP_SIZE,
        "resolution": COSTMAP_RES,
        "robot_radius": ROBOT_RADIUS,
        "max_range": MAX_RANGE,
        "estop_active": estop_active,
        "last_update": last_update_time,
    }


@app.post("/update_depth")
async def update_depth(req: UpdateDepthRequest):
    try:
        img_bytes = base64.b64decode(req.image)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        depth_img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise ValueError("Could not decode depth image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    depth_to_costmap(depth_img, req.fx, req.fy, req.cx, req.cy)

    return {
        "status": "ok",
        "estop_active": estop_active,
        "costmap_coverage": float(np.mean(costmap > 0.1)),
        "timestamp": last_update_time,
    }


@app.post("/safe_velocity")
async def safe_velocity(req: SafeVelocityRequest):
    if estop_active:
        return {
            "safe_vx": 0.0,
            "safe_omega": 0.0,
            "estop_active": True,
            "reason": "Emergency stop — obstacle too close",
        }

    vx, omega, info = dwa_plan(req.target_vx, 0.0, req.target_omega, req.goal_x, req.goal_y)

    return {
        "safe_vx": vx,
        "safe_omega": omega,
        "estop_active": False,
        "dwa_score": info.get("score", 0.0),
    }


@app.get("/costmap")
async def get_costmap(format: str = "png"):
    if costmap is None:
        raise HTTPException(status_code=503, detail="Costmap not initialized")

    if format == "json":
        return {"costmap": costmap.tolist(), "size": COSTMAP_SIZE, "resolution": COSTMAP_RES}

    # Return as PNG image
    vis = (costmap * 255).astype(np.uint8)
    # Draw robot position
    c = COSTMAP_SIZE // 2
    cv2.circle(vis, (c, c), max(1, int(ROBOT_RADIUS / COSTMAP_RES)), 128, 1)
    _, buf = cv2.imencode(".png", vis)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/estop_status")
async def estop_status():
    return {
        "estop_active": estop_active,
        "min_obstacle_distance_threshold": MIN_OBSTACLE_DIST,
        "last_update": last_update_time,
    }
