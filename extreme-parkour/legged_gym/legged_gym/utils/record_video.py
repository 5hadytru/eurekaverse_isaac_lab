import imageio
import os
import gymnasium as gym
import torch

class MultiCamVideo(gym.Wrapper):
    def __init__(self, env, out_dir, cam_name_to_env_ids: dict, fps=30, length=float("inf")):
        super().__init__(env)
        self.out_dir, self.fps = out_dir, fps
        self.len = length
        self.episode = 0
        os.makedirs(out_dir, exist_ok=True)
        self.cam_name_to_env_ids = cam_name_to_env_ids
        self.writers = {cam_name: imageio.get_writer(f"{self.out_dir}/{cam_name}.mp4", fps=self.fps)
                        for cam_name in cam_name_to_env_ids.keys()}
        self.frame = 0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if self.frame < self.len:
            for cam_name in self.cam_name_to_env_ids.keys():
                # Get the image tensor (X, H, W, 3)
                image = self.env.scene["cam_" + cam_name].data.output["rgb"][self.cam_name_to_env_ids[cam_name]]
                print(f"Received {image.shape} from {cam_name}")
                
                # Arrange images in 2x3 grid
                if len(image.shape) == 4:  # (X, H, W, 3)
                    X, H, W, C = image.shape
                    if X == 6:  # Ensure we have exactly 6 images for 2x3 grid
                        # Reshape to (2, 3, H, W, 3) then rearrange to create grid
                        grid_images = image.view(2, 3, H, W, C)
                        # Concatenate: first along width (dim 3), then along height (dim 1)
                        grid_image = torch.cat([
                            torch.cat([grid_images[0, 0], grid_images[0, 1], grid_images[0, 2]], dim=1),  # Top row
                            torch.cat([grid_images[1, 0], grid_images[1, 1], grid_images[1, 2]], dim=1)   # Bottom row
                        ], dim=0)  # Stack rows vertically
                        # Convert to numpy and write to video
                        self.writers[cam_name].append_data(grid_image.cpu().numpy().astype('uint8'))
                    else:
                        print(f"Warning: Expected 6 images for 2x3 grid, got {X} images for camera {cam_name}")
                else:
                    print(f"Warning: Unexpected image shape {image.shape} for camera {cam_name}")
        
        self.frame += 1
        return obs, r, term, trunc, info
    
    def close(self):
        # Close all video writers
        for writer in self.writers.values():
            writer.close()
        super().close()