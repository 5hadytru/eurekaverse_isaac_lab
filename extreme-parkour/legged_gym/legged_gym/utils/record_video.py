import imageio
import os
import gymnasium as gym

class MultiCamVideo(gym.Wrapper):
    def __init__(self, env, out_dir, cam_names:list, fps=30, length=float("inf")):
        super().__init__(env)
        self.out_dir, self.fps = out_dir, fps
        self.len = length
        self.episode = 0
        os.makedirs(out_dir, exist_ok=True)
        self.cam_names = cam_names
        self.writers = {cam_name: imageio.get_writer(f"{self.out_dir}/{cam_name}.mp4", fps=self.fps)
                        for cam_name in cam_names}
        self.frame = 0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if self.frame < self.len:
            for c_i, c in enumerate(self.cam_names):
                image = self.env.scene["cam_" + c].data.output["rgb"]
                # self.writers[c].append_data(image)
        self.frame += 1
        
        return obs, r, term, trunc, info
