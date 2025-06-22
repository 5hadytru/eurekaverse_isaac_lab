import imageio
import os
import gymnasium as gym
import torch


class MultiCamVideo(gym.Wrapper):
    def __init__(self, env, out_dir, cam_name_to_env_ids: dict, fps=30, length=float("inf")):
        super().__init__(env)
        self.out_dir, self.fps = out_dir, fps
        self.len = length
        os.makedirs(out_dir, exist_ok=True)
        self.cam_name_to_env_ids = cam_name_to_env_ids
        self.writers = {
            cam_name: imageio.get_writer(f"{self.out_dir}/{cam_name}.mp4", fps=self.fps)
            for cam_name in cam_name_to_env_ids.keys()
        }
        self.frame = 0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        if self.frame < self.len:
            for cam_name, env_ids in self.cam_name_to_env_ids.items():
                # Retrieve the batch of images for this camera: shape (X, H, W, 3)
                image_batch = self.env.scene[cam_name].data.output["rgb"][0]
                # image_batch = self.env.scene[cam_name].data.output["rgb"][env_ids]

                print(f"Received {image_batch.shape} from {cam_name}")

                if len(image_batch.shape) == 4:  # Expecting (X, H, W, 3)
                    X, H, W, C = image_batch.shape
                    if X >= 1:
                        # Take only the first image in the batch
                        first_image = image_batch[0]  # shape (H, W, 3)
                        # Convert to numpy uint8 and write directly
                        self.writers[cam_name].append_data(first_image.cpu().numpy().astype("uint8"))
                    else:
                        raise Exception(
                            f"No images found (X={X}) for camera '{cam_name}', cannot index 0."
                        )
                else:
                    raise Exception(
                        f"Unexpected image shape {image_batch.shape} for camera '{cam_name}'."
                    )

        self.frame += 1
        return obs, r, term, trunc, info

    def close(self):
        # Close all video writers and then the underlying environment
        for writer in self.writers.values():
            writer.close()
        super().close()


def get_camera_coords(col_idx, row_idx, cam_height=5.75):
    """
    Get camera position and rotation parameters for a specific terrain cell.
    
    Args:
        col_idx: Column index in the terrain grid
        row_idx: Row index in the terrain grid
        
    Returns:
        dict: Camera configuration with position and rotation
    """

    all_configs = {
        (tuple(range(8)), 0): ((-13.5, 11.5, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((0,1), 0): ((-13.5, 11.5, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((2,3), 0): ((-3.5, 19.5, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((0,1), 1): ((-1.5, 6.5, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((2,3), 1): ((8.5, 14.25, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((0,1), 2): ((10.75, 1.25, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((2,3), 2): ((20.5, 9.5, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((0,1), 3): ((22.6, -3.5, cam_height), (1.0, 0.0, 0.25, 0.0)),
        # ((2,3), 3): ((32.75, 4.5, cam_height), (1.0, 0.0, 0.25, 0.0)),
    }

    return {
        "position": all_configs[(col_idx, row_idx)][0],
        "rotation": all_configs[(col_idx, row_idx)][1]
    }
    
    return camera_config