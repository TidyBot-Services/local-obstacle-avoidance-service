"""
Local Base Obstacle Avoidance — Python Client SDK

Usage (inside robot code execution):
    from service_clients.local_obstacle_avoidance.client import ObstacleAvoidanceClient
    from robot_sdk import sensors

    client = ObstacleAvoidanceClient()

    # Update costmap with depth frame
    depth_bytes = sensors.get_depth_frame()
    result = client.update_depth(depth_bytes, fx=615.0, fy=615.0, cx=320.0, cy=240.0)
    print(result["estop_active"])

    # Get safe velocity command
    cmd = client.safe_velocity(target_vx=0.3, target_omega=0.0)
    print(f"Safe: vx={cmd['safe_vx']:.2f}, omega={cmd['safe_omega']:.2f}")

    # Check e-stop
    estop = client.estop_status()
    print(f"E-stop active: {estop['estop_active']}")

    # Get costmap as PNG bytes
    costmap_png = client.costmap(format="png")
"""

import base64
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import numpy as np


class ObstacleAvoidanceClient:
    """Client SDK for the Local Base Obstacle Avoidance service."""

    def __init__(self, host: str = "http://158.130.109.188:8007", timeout: float = 30.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """Check if the service is running."""
        req = urllib.request.Request(f"{self.host}/health")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _encode_image(self, image) -> str:
        """Encode image to base64 from bytes, file path, numpy array, or pass through."""
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        elif isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, np.ndarray):
            import cv2
            _, buf = cv2.imencode(".png", image)
            return base64.b64encode(buf.tobytes()).decode()
        return image  # assume already base64

    def _post(self, endpoint: str, payload: dict) -> dict:
        """POST JSON to service and return parsed response."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{endpoint} failed (HTTP {e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Service unavailable at {self.host}: {e.reason}") from e

    def _get(self, endpoint: str) -> dict:
        """GET JSON from service."""
        req = urllib.request.Request(f"{self.host}{endpoint}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{endpoint} failed (HTTP {e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Service unavailable at {self.host}: {e.reason}") from e

    def update_depth(self, image, fx: float, fy: float, cx: float, cy: float) -> dict:
        """
        Submit a depth frame to update the local costmap.

        Args:
            image: Depth image as bytes (PNG), file path (str), numpy array, or base64 string.
            fx: Camera focal length x (pixels).
            fy: Camera focal length y (pixels).
            cx: Camera principal point x (pixels).
            cy: Camera principal point y (pixels).

        Returns:
            dict with keys: status, estop_active, costmap_coverage, timestamp
        """
        payload = {
            "image": self._encode_image(image),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        }
        return self._post("/update_depth", payload)

    def safe_velocity(self, target_vx: float = 0.0, target_omega: float = 0.0,
                      goal_x: Optional[float] = None, goal_y: Optional[float] = None) -> dict:
        """
        Get safe velocity command avoiding obstacles.

        Args:
            target_vx: Desired forward velocity (m/s).
            target_omega: Desired angular velocity (rad/s).
            goal_x: Optional goal x in robot frame (meters, forward).
            goal_y: Optional goal y in robot frame (meters, left).

        Returns:
            dict with keys: safe_vx, safe_omega, estop_active, dwa_score
        """
        payload = {"target_vx": target_vx, "target_omega": target_omega}
        if goal_x is not None:
            payload["goal_x"] = goal_x
        if goal_y is not None:
            payload["goal_y"] = goal_y
        return self._post("/safe_velocity", payload)

    def costmap(self, format: str = "png"):
        """
        Retrieve the current local costmap.

        Args:
            format: "png" returns raw PNG bytes, "json" returns dict with costmap array.

        Returns:
            bytes (PNG) if format="png", dict if format="json".
        """
        if format == "json":
            return self._get("/costmap?format=json")
        # PNG: return raw bytes
        req = urllib.request.Request(f"{self.host}/costmap?format=png")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"/costmap failed (HTTP {e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Service unavailable at {self.host}: {e.reason}") from e

    def estop_status(self) -> dict:
        """
        Check if emergency stop is active.

        Returns:
            dict with keys: estop_active, min_obstacle_distance_threshold, last_update
        """
        return self._get("/estop_status")
