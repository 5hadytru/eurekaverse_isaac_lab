import imageio
import os
import gymnasium as gym

class MultiCamVideo(gym.Wrapper):
    def __init__(self, env, out_dir, fps=30, length=float("inf")):
        super().__init__(env)
        self.out_dir, self.fps = out_dir, fps
        self.len = int(length)
        self.episode = 0
        os.makedirs(out_dir, exist_ok=True)

    def reset_cameras(self, cam_ids):
        obs, info = self.env.reset(*a, **kw)
        self.writers = [imageio.get_writer(f"{self.out_dir}/cam{i}_ep{self.episode}.mp4",
                                           fps=self.fps) 
                        for i in range(self.env.tcam.num_cameras)]
        self.frame = 0; self.episode += 1
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if self.frame < self.len:
            imgs = self.env.tcam.data.output["rgb"]      # (N,H,W,3)
            for img, w in zip(imgs, self.writers):
                w.append_data(img)
        self.frame += 1
        if term or trunc:
            for w in self.writers: w.close()
        return obs, r, term, trunc, info

    
    