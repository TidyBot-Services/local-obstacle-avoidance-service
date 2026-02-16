# Local Base Obstacle Avoidance Service

Real-time local obstacle avoidance for TidyBot. Takes depth images, maintains a local 2D costmap (~3–5m range), and provides safe velocity commands using the Dynamic Window Approach (DWA). Pure Python — no ROS needed.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/update_depth` | Submit depth frame + camera intrinsics → updates local costmap |
| POST | `/safe_velocity` | Given target velocity → returns safe velocity avoiding obstacles |
| GET | `/costmap` | Retrieve current local costmap (PNG image or JSON array) |
| GET | `/estop_status` | Check if emergency stop is active (obstacle too close) |
| GET | `/health` | Service health check |

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8007
```

## Client Usage

```python
from client import ObstacleAvoidanceClient

client = ObstacleAvoidanceClient()

# Update costmap with depth frame
result = client.update_depth(depth_bytes, fx=615.0, fy=615.0, cx=320.0, cy=240.0)

# Get safe velocity
cmd = client.safe_velocity(target_vx=0.3, target_omega=0.0)
print(f"vx={cmd['safe_vx']:.2f}, omega={cmd['safe_omega']:.2f}")

# Check e-stop
print(client.estop_status())
```

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOT_RADIUS` | 0.18 | Robot radius in meters |
| `MIN_OBSTACLE_DIST` | 0.25 | E-stop threshold in meters |
| `MAX_RANGE` | 4.0 | Max depth range in meters |
| `COSTMAP_SIZE` | 200 | Grid cells per side |
| `COSTMAP_RES` | 0.025 | Meters per cell |
| `DWA_MAX_VEL` | 0.5 | Max forward velocity (m/s) |
| `DWA_MAX_OMEGA` | 1.5 | Max angular velocity (rad/s) |
