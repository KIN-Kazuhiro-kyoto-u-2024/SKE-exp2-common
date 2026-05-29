import numpy as np
from dataclasses import dataclass
import os
import pathlib
import sys

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))

from env import make_env
from models import Qtable
from utils.logger import Logger
from collections import OrderedDict

@dataclass
class EnvConfig:
    domain: str = "double_pendulum"
    task: str = "balance"
    num_digitized: int = 16
    num_action: int = 9
    state_size: int = num_digitized**4
    gamma: float = 0.99
    alpha: float = 0.5
    max_episode: int = int(1)
    episode_length: int = 400
    should_log_model: int = 10000
    should_log_scalar: int = 100
    should_log_video: int = 10000
    restore: bool = False
    restore_path: str = "./logs/10-12-12-54-05/qtable_100000.npy"
    video_length: int = 400
    logdir: str = "./logs/eval/"

class Agent():
    def __init__(self, config: EnvConfig) -> None:
        self._config = config
        self._build_model()

    def get_action(self, state, global_step=None, explore=True, method="softmax"):
        return self._qtable.get_action(state, explore, global_step, method=method)

    def update_Qtable(self, state, action, reward, next_state):
        return self._qtable.update_Qtable(state, action, reward, next_state)

    def _build_model(self):
        self._qtable = Qtable(self._config)
    
def main():
    config = EnvConfig()
    env = make_env(config)
    os.makedirs(config.logdir, exist_ok=True)
    logger = Logger(config.logdir)
    logger.log_config(config)
    agent = Agent(config)

    print(config.restore_path)
    agent._qtable.load(config.restore_path)

    # main training loop
    for episode in range(config.max_episode):

        # １エピソードの実行
        state = env.reset()
        episode_reward = 0
        act_seq = []
        img_seq = []
        rew_seq = []
        alpha_seq = []
        theta_seq = []
        n_alpha_seq = []
        n_theta_seq = []
        for step in range(config.episode_length):
            img_seq.append(env._env.physics.render(height=480, width=640,camera_id=0))
            action = agent.get_action(state, explore=False)
            act_seq.append(env._digitized_action[action])
            next_state, reward, done, state_dict = env.step(action)
            rew_seq.append(reward)
            alpha_seq.append(state_dict['pendulum_rad'])
            theta_seq.append(state_dict['arm_rad'])
            n_alpha_seq.append(state_dict['n_pendulum_rad'])
            n_theta_seq.append(state_dict['n_arm_rad'])
            episode_reward += reward
            state = next_state
            if done:
                break
        # スカラーをログに保存
        print(f"\nepisode: {episode}, episode_reward: {episode_reward}, episode_steps: {step}")
        logger.add_scalars(OrderedDict([
            ("episode_reward", episode_reward),
        ]))

        act_seq_img = logger.plot2image('action', {'torque': np.array(act_seq)})
        rew_seq_img = logger.plot2image('reward', {'reward': np.array(rew_seq)})
        state_seq_img = logger.plot2image('state', {'alpha': np.array(alpha_seq),
                                                    'theta': np.array(theta_seq)})
        n_state_seq_img = logger.plot2image('n_state', {'n_alpha': np.array(n_alpha_seq),
                                                        'n_theta': np.array(n_theta_seq)})
        img_dict = {'action': act_seq_img, 'reward': rew_seq_img,
                    'state': state_seq_img, 'n_state': n_state_seq_img}
        logger.log_image(img_dict)
        logger.log_video({'video': np.array(img_seq)}, save=False)
        logger.step()

if __name__ == "__main__":
    main()

