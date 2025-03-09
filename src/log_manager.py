import os
import cv2
import base64
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
class LogManager:
    """Manages logging to markdown files and UI updates for the VLN agent."""
    def __init__(self, log_dir: str, web_ui):
        self.log_dir = log_dir
        self.web_ui = web_ui
        self.current_data: Dict[str, Any] = {}
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def start_episode(self, episode_idx: int, instruction: str) -> None:
        """Initialize logging for a new episode."""
        episode_folder = os.path.join(self.log_dir, f"{self.current_time}_episode_{episode_idx}")
        os.makedirs(episode_folder, exist_ok=True)
        md_file = os.path.join(episode_folder, "log.md")
        # with open(md_file, 'w') as f:
        #     f.write(f"# Episode {episode_idx}\n\n")
        #     f.write(f"**Instruction:** {instruction}\n\n")
        
        # Initialize UI data with latest_depth_image
        self.current_data = {
            'episode_idx': episode_idx,
            'instruction': instruction,
            'step_counter': 0,
            'robot_path': [],
            'current_action': 'Stop moving',
            'rgb_image': '',
            'latest_depth_image': '',  # Added for real-time depth map
            'goal_distance': 0.0,
            'status': 'Running'
        }
        self.web_ui.update_data(self.current_data)

    def log_step(self, step_counter: int, robot_path: List[tuple], rgb_image_np: np.ndarray, depth_map: np.ndarray, current_action: str, goal_distance: float, remaining_steps: int) -> None:
        """Log and update UI with step-specific data, including depth map."""
        goal_distance_serializable = float(goal_distance.item()) if hasattr(goal_distance, 'item') else float(goal_distance)
        robot_path_serializable = [
            [float(x.item() if hasattr(x, 'item') else x), float(y.item() if hasattr(y, 'item') else y)]
            for x, y in robot_path
        ]
        # Resize images to 256x256 for efficiency
        resized_rgb = cv2.resize(rgb_image_np, (256, 256))
        resized_depth = cv2.resize(depth_map, (256, 256), interpolation=cv2.INTER_NEAREST)
        depth_colored = self.depth_to_colormap(resized_depth)
        latest_depth_image = self.image_to_base64(depth_colored)
        
        self.current_data.update({
            'step_counter': step_counter,
            'remaining_steps': remaining_steps,
            'robot_path': robot_path_serializable,
            'current_action': current_action,
            'goal_distance': goal_distance_serializable,
            'rgb_image': self.image_to_base64(resized_rgb),
            'latest_depth_image': latest_depth_image  # Added for real-time depth
        })
        self.web_ui.update_data(self.current_data)

    def set_status(self, status: str) -> None:
            """Update the episode status and send to UI."""
            self.current_data['status'] = status
            self.web_ui.update_data(self.current_data)
    def depth_to_colormap(self, depth_np: np.ndarray) -> np.ndarray:
        """Convert depth map to colorized image for logging."""
        depth_norm = np.clip(depth_np, 0.5, 5.0)
        depth_norm = (depth_norm - 0.5) / 4.5 * 255
        depth_norm = depth_norm.astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    def image_to_base64(self, img_np) -> str:
        """Convert numpy image to base64 string for UI."""
        _, buffer = cv2.imencode('.jpg', img_np)
        return base64.b64encode(buffer).decode('utf-8')
    def end_episode(self, measurements: Dict[str, Any]) -> None:
        """Finalize logging with episode measurements."""
        episode_folder = os.path.join(self.log_dir, f"episode_{self.current_data['episode_idx']}")
        md_file = os.path.join(episode_folder, "log.md")
        # with open(md_file, 'a') as f:
        #     f.write("## Episode Measurements\n\n")
        #     for key, value in measurements.items():
        #         f.write(f"- **{key}:** {value}\n")
        #     f.write("\n")

    @staticmethod
    def image_to_base64(img_np) -> str:
        """Convert numpy image to base64 string for UI."""
        _, buffer = cv2.imencode('.jpg', img_np)
        return base64.b64encode(buffer).decode('utf-8')