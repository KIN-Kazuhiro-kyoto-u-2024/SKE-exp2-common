from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

class Logger():
    def __init__(self, log_dir, start=0) -> None:
        self.log_dir = log_dir

        self.global_step = start
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_config(self, config):
        self.writer.add_text('config', str(config), 0)
    
    def add_scalars(self, scalars_dict):
        for key, value in scalars_dict.items():
            self.writer.add_scalar(key, value, self.global_step)
        self.writer.flush()
    
    def step(self):
        self.global_step += 1
    
    def log_video(self, video_dict, fps=30, save=True):
        if save:
            video = list(video_dict.values())[0]
            video_path = self.log_dir + f"video_{self.global_step}.mp4"
            h,w = video.shape[1:3]
            print(f"output video path: {video_path}")
            out = cv2.VideoWriter(filename=video_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(w,h))
            for t in range(video.shape[0]):
                out.write(video[t])
            out.release()
        for key, value in video_dict.items():
            if value.shape[-1] == 3:
                value = value.transpose(0, 3, 1, 2)
            value = value[:, ::-1]
            self.writer.add_video(key, value[np.newaxis], self.global_step)

    def log_image(self, img_dict):
        for key, value in img_dict.items():
            if value.shape[-1] == 3:
                value = value.transpose(2, 0, 1)
            self.writer.add_image(key, value, self.global_step)
        self.writer.flush()

    def record_csv(self, x_time, y_shoulder_rad, y_elbow_rad, name):
        with open(self.log_dir+name+'.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(x_time)
            writer.writerow(y_shoulder_rad)
            writer.writerow(y_elbow_rad)

    def plot2image(self, tag, plot_data, time=None):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_xlabel('t')
        ax.set_ylabel('value')
        ax.set_title(tag)
        if time is None:
            t = np.arange(list(plot_data.values())[0].shape[0])
        else:
            t = time
        for key, value in plot_data.items():
            ax.plot(t, value, label=key)
        fig.tight_layout()
        fig.legend(loc='upper right')
        plt.grid()
        fig.canvas.draw()
        plot_image = np.array(fig.canvas.renderer._renderer)[..., :3]
        plt.clf()
        plt.close()
        return plot_image