from dataclasses import dataclass
import time
import pathlib

import numpy as np
import control as ctrl

from utils.logger import Logger
from real.invpen import Invpen
from rl.env import make_env

@dataclass
class EnvConfig:
    domain: str = "double_pendulum"
    task: str = "sys_iden"
    init_alpha: float = np.pi
    repeat: int = 2

    # パラメータ同定用のデータを保存するフォルダ名をここに記載
    logdir: str = pathlib.Path().joinpath("./logs/real-for-sys-iden/static_fric", str(time.strftime("%m-%d-%H-%M-%S")))

    #以下はパラメータ同定時には関係なし
    num_digitized: int = 8
    num_action: int = 4
    state_size: int = num_digitized**4
    gamma: float = 0.99
    alpha: float = 0.5
    max_episode: int = int(10e7)
    episode_length: int = 500
    should_log_model: int = 10000
    should_log_scalar: int = 200
    should_log_video: int = 10000
    restore: bool = False
    restore_path: str = ""
    video_length: int = 200

class Agent():

    def __init__(self, eps):

        config = EnvConfig()
        self.config = config
        self.logger = Logger(config.logdir)
        self.env = make_env(config)
        self.start = False
        self.data = {'n_alpha': [], 'n_theta': []}
        self.sim_data = {'alpha': [], 'theta': [], 'n_alpha': [], 'n_theta': [], 'torque': [0], 'dt': [0]}
        state_dict = self.env.reset()
        self.sim_data['alpha'].append(state_dict['alpha'])
        self.sim_data['theta'].append(state_dict['theta'])

        # トルク入力の初期値
        self.torque = 0.1

    def policy(self, state):

        torque = self.torque
        self.start = True

        # アームが回転しすぎたとき，非常停止
        if state[0] > 0.5*np.pi:
            torque  = 0
        
        state_dict, _, _, _ = self.env.step(torque)
        alpha = state_dict['alpha']
        self.sim_data['alpha'].append(alpha)
        self.sim_data['theta'].append(state_dict['theta'])
        self.sim_data['torque'].append(torque)
        self.sim_data['dt'].append(0.02)

        # トルクの値を徐々に増加させる（必要であれば）
        # self.torque += 0.001
        
        return torque

    def after_termination_func(self, data):
        data = {key: np.array(value) for key, value in data.items()}
        sim_data = {key: np.array(value) for key, value in self.sim_data.items()}
        np.save(EnvConfig().logdir.joinpath('data.npy'), data)
        np.save(EnvConfig().logdir.joinpath('sim_data.npy'), sim_data)
        alpha_f = data['alpha']
        alpha_sim_f = sim_data['alpha']
        for i in range(len(data['alpha'])):
            if alpha_f[i] > 0:
                alpha_f[i] = np.pi - alpha_f[i]
            else:
                alpha_f[i] = -np.pi - alpha_f[i]

            if alpha_sim_f[i] > 0:
                alpha_sim_f[i] = np.pi - alpha_sim_f[i]
            else:
                alpha_sim_f[i] = -np.pi - alpha_sim_f[i]
        state_seq_img = self.logger.plot2image('state', {
            'alpha':data['alpha'],
            'alpha_f':alpha_f,
            'theta':data['theta'],
            }, np.cumsum(data['dt']))
        act_seq_img = self.logger.plot2image('torque', {'torque': data['torque']}, np.cumsum(data['dt']))
        time_seq_img = self.logger.plot2image('time', {'time': data['time']})
        dt_seq_img = self.logger.plot2image('dt', {'dt': data['dt']})
        img_dict = {'action/real': act_seq_img,
                    'state/real': state_seq_img,
                    'time/real': time_seq_img,
                    'dt/real': dt_seq_img}

        state_seq_img = self.logger.plot2image('state', {
            'alpha': sim_data['alpha'],
            'alpha_f': alpha_sim_f,
            'theta': sim_data['theta'],
            }, np.cumsum(sim_data['dt']))
        act_seq_img = self.logger.plot2image('action', {'torque': sim_data['torque']}, np.cumsum(sim_data['dt']))
        time_seq_img = self.logger.plot2image('time', {'time': np.cumsum(sim_data['dt'])})
        dt_seq_img = self.logger.plot2image('dt', {'dt': sim_data['dt']})
        sim_img_dict = {'action/sim': act_seq_img,
                    'state/sim': state_seq_img,
                    'time/sim': time_seq_img,
                    'dt/sim': dt_seq_img}

        self.logger.log_image(img_dict)
        self.logger.log_image(sim_img_dict)


def main():
    agent = Agent(eps=np.pi/15.0)
    invpen = Invpen(agent)

    invpen.run(sample_time=0.020, simulation_time=1.0, figure=True, logging=True)

if __name__ == "__main__":
    main()