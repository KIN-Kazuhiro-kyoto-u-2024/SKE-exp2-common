from collections import OrderedDict

import numpy as np

from simulator.acrobot import balance, swingup, sys_iden


def make_env(config):
    if config.domain == "double_pendulum":
        if config.task == "swingup":
            return SwingUp(config)
        elif config.task == "balance":
            return Balance(config)
        elif config.task == "sys_iden":
            return SysIden(config)
    raise NotImplementedError


class Balance:

    def __init__(self, config):

        self._config = config
        d = config.num_digitized
        self._env = balance()
        self._pendulum_limit = (
            -0.20 * np.pi,
            0.20 * np.pi,
        )  # alpha の範囲．これを超えたら倒立維持失敗．
        self._arm_limit = (
            -0.9 * np.pi,
            0.9 * np.pi,
        )  # theta の範囲．これを超えたら倒立維持失敗．

        self._arrange = 0  # 実機用：SwingUpでは零行列だったので0でいいはず

        print("dt = ", config.repeat * self._env.control_timestep())
        self._digitized_action = np.linspace(
            self._env.action_spec().minimum[0],
            self._env.action_spec().maximum[0],
            config.num_action,
        )
        print("action interval: ", self._digitized_action)
        print("state-action is ", f"({d}x{d}x{d}x{d}) x {config.num_action}")
        print(f"total state-action number is {(d**2)*((d)**2)*config.num_action}")

    @property
    def action_space(self):
        return self._config.num_action

    @property
    def obs_space(self):
        return self._config.state_size

    def step(self, action, time=None, timedepedence=None):
        action = self._digitized_action[action]
        for _ in range(self._config.repeat):
            obs = self._env.step(action).observation
        digitized_state, done, state_dict = self._digitized_state(obs)
        reward, best = self._get_reward(state_dict, done, action)
        state_dict["best"] = best
        return digitized_state, reward, done, state_dict

    def reset(self):
        obs = self._env.reset().observation
        digitized_state, _, state_dict = self._digitized_state(obs)
        return digitized_state

    def _get_reward(self, state_dict, done, action):

        # 倒立状態の維持に失敗したときの報酬
        if done:
            return -10, 0

        # 報酬（rew）の設定
        # rew の与え方を色々変更してみる
        d = self._config.num_digitized
        n_pendulum_rad, n_pendulum_vel = (
            state_dict["n_pendulum_rad"],
            state_dict["n_pendulum_vel"],
        )
        n_best = (d - 1) / 2
        n_arm_rad, n_arm_vel = state_dict["n_arm_rad"], state_dict["n_arm_vel"]
        n_arm_best = (d - 1) / 2
        bonus = 0
        rew = 0.1
        return rew, 1 if bonus > 0 else 0

    def _digitized_state(self, obs):

        # シミュレータの出力結果から theta, thetadot, alpha, alphadot の値を取得
        state_dict = OrderedDict()
        d = self._config.num_digitized
        vec, vel = obs["orientations"], obs["velocity"]
        arm_vec, pendulum_vec = vec[0:2], vec[2:4]
        arm_vel, pendulum_vel = vel[0], vel[1]
        arm_rad = np.arctan2(arm_vec[1], arm_vec[0])
        pendulum_rad = np.arctan2(pendulum_vec[1], pendulum_vec[0])

        # arm_rad: theta, arm_vel: thetadot, pendulum_rad: alpha, pendulum_vel: alphadot
        # 以下を書き換えて，これら（連続値）の離散化方法を色々変更してみる
        n_arm_rad = np.digitize(
            arm_rad, np.linspace(self._arm_limit[0], self._arm_limit[1], d + 1)[1:-1]
        )
        n_arm_vel = np.digitize(arm_vel.clip(-8, 8), np.linspace(-8, 8, d + 1)[1:-1])
        n_pendulum_rad = np.digitize(
            pendulum_rad,
            np.linspace(self._pendulum_limit[0], self._pendulum_limit[1], d + 1)[1:-1],
        )
        n_pendulum_vel = np.digitize(
            pendulum_vel.clip(-8, 8), np.linspace(-8, 8, d + 1)[1:-1]
        )

        state_dict["arm_rad"] = arm_rad
        state_dict["pendulum_rad"] = pendulum_rad
        state_dict["arm_vel"] = arm_vel
        state_dict["pendulum_vel"] = pendulum_vel
        state_dict["n_arm_rad"] = n_arm_rad
        state_dict["n_pendulum_rad"] = n_pendulum_rad
        state_dict["n_arm_vel"] = n_arm_vel
        state_dict["n_pendulum_vel"] = n_pendulum_vel
        state_dict["digitized_state"] = (
            n_pendulum_rad + n_pendulum_vel * d + n_arm_vel * d**2 + n_arm_rad * d**3
        )

        # theta か alpha が既定の範囲を超えたら（倒立状態の維持に失敗したら）
        # done を True としてエピソード終了
        arm_cond = n_arm_rad == 0 or n_arm_rad == d - 1
        pendulum_cond = n_pendulum_rad == 0 or n_pendulum_rad == d - 1
        if arm_cond or pendulum_cond:
            done = True
        else:
            done = False
        return state_dict["digitized_state"], done, state_dict


####################################
## SysIden: システム同定用の処理 #####
####################################


class SysIden:
    def __init__(self, config):
        self._config = config
        self._env = sys_iden(init_alpha=config.init_alpha)
        print("dt = ", config.repeat * self._env.control_timestep())

    @property
    def action_space(self):
        return self._env.action_spec()

    @property
    def obs_space(self):
        return self._env.observation_spec()

    def step(self, action, time=None, timedepedence=None):
        for _ in range(self._config.repeat):
            obs = self._env.step(action).observation
        state_dict, done = self._get_obs(obs)
        reward = 0
        return state_dict, reward, done, state_dict

    def reset(self):
        obs = self._env.reset().observation
        state_dict, done = self._get_obs(obs)
        return state_dict

    def _get_obs(self, obs):
        # 0 is positive z axis at doule pendulum pendulum angle
        state_dict = OrderedDict()
        vec, vel = obs["orientations"], obs["velocity"]
        arm_vec, pendulum_vec = vec[0:2], vec[2:4]
        arm_vel, pendulum_vel = vel[0], vel[1]
        arm_rad = np.arctan2(arm_vec[1], arm_vec[0])
        pendulum_rad = np.arctan2(pendulum_vec[1], pendulum_vec[0])

        state_dict["theta"] = arm_rad
        state_dict["alpha"] = pendulum_rad
        state_dict["theta_dot"] = arm_vel
        state_dict["alpha_dot"] = pendulum_vel
        done = False
        return state_dict, done

    @property
    def _digitized_action(self):
        raise NotImplementedError


#####################################
## 以下は共通課題では不要 #############
#####################################


class SwingUp:
    def __init__(self, config):
        self._config = config
        self._arrange = np.zeros(config.num_digitized - 1)
        self._env = swingup()
        print("dt = ", self._env.control_timestep())
        # self._arrange[config.num_digitized//2] = -0.20
        # self._arrange[config.num_digitized//2 - 2] = 0.20
        self._digitized_action = np.linspace(
            self._env.action_spec().minimum[0],
            self._env.action_spec().maximum[0],
            config.num_action,
        )
        print(
            "alpha interval: ",
            np.linspace(-np.pi, np.pi, config.num_digitized + 1)[1:-1] + self._arrange,
        )
        print(
            "theta interval: ",
            np.linspace(-np.pi, np.pi, config.num_digitized + 1)[1:-1],
        )
        d = config.num_digitized
        print("action interval: ", self._digitized_action)
        print("state-action is ", f"({d}x{d}x{d}x{d}) x {config.num_action}")
        print(f"total state-action number is {(d**2)*((d)**2)*config.num_action}")

    @property
    def action_space(self):
        return self._config.num_action

    @property
    def obs_space(self):
        return self._config.state_size

    def step(self, action, time=None, timedepedence=None):
        action = self._digitized_action[action]
        for _ in range(self._config.repeat):
            obs = self._env.step(action).observation
        digitized_state, done, state_dict = self._digitized_state(obs)
        reward, best = self._get_reward(state_dict, done)
        state_dict["best"] = best
        return digitized_state, reward, done, state_dict

    def reset(self):
        obs = self._env.reset().observation
        digitized_state, _, state_dict = self._digitized_state(obs)
        return digitized_state

    def _get_reward(self, state_dict, done):
        raise NotImplementedError

    def _digitized_state(self, obs):
        # 0 is positive z axis at doule pendulum pendulum angle
        state_dict = OrderedDict()
        d = self._config.num_digitized
        vec, vel = obs["orientations"], obs["velocity"]
        arm_vec, pendulum_vec = vec[0:2], vec[2:4]
        arm_vel, pendulum_vel = vel[0], vel[1]
        arm_rad = np.arctan2(arm_vec[1], arm_vec[0])
        pendulum_rad = np.arctan2(pendulum_vec[1], pendulum_vec[0])
        n_arm_rad = np.digitize(arm_rad, np.linspace(-np.pi, np.pi, d + 1)[1:-1])
        n_arm_vel = np.digitize(arm_vel.clip(-8, 8), np.linspace(-8, 8, d + 1)[1:-1])

        n_pendulum_rad = np.digitize(
            pendulum_rad, np.linspace(-np.pi, np.pi, d + 1)[1:-1] + self._arrange
        )
        n_pendulum_vel = np.digitize(
            pendulum_vel.clip(-8, 8), np.linspace(-8, 8, d + 1)[1:-1]
        )

        state_dict["arm_rad"] = arm_rad
        state_dict["pendulum_rad"] = pendulum_rad
        state_dict["arm_vel"] = arm_vel
        state_dict["pendulum_vel"] = pendulum_vel
        state_dict["n_arm_rad"] = n_arm_rad
        state_dict["n_pendulum_rad"] = n_pendulum_rad
        state_dict["n_arm_vel"] = n_arm_vel
        state_dict["n_pendulum_vel"] = n_pendulum_vel
        state_dict["digitized_state"] = (
            n_pendulum_rad + n_pendulum_vel * d + n_arm_vel * d**2 + n_arm_rad * d**3
        )
        if n_arm_rad == 0 or n_arm_rad == d - 1:
            done = True
        else:
            done = False
        return state_dict["digitized_state"], done, state_dict


def main():
    import time
    from dataclasses import dataclass

    from PIL import Image

    @dataclass
    class EnvConfig:
        domain: str = "double_pendulum"
        task: str = "swingup"
        num_digitized: int = 16
        num_action: int = 2
        state_size: int = num_digitized**3
        gamma: float = 0.99
        alpha: float = 0.5
        max_episode: int = int(10e3)
        episode_length: int = 400
        should_log_model: int = int(10e3)
        should_log_scalar: int = int(10)
        should_log_video: int = int(50)
        restore: bool = False
        restore_file: str = "Qtable.npy"
        video_length: int = 400
        logdir: str = "./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "/"

    env = SwingUp(EnvConfig())
    # env = suite.load(domain_name="acrobot", task_name="swingup")
    env.reset()
    for i in range(100):
        img = Image.fromarray(
            env._env.physics.render(height=480, width=640, camera_id=0)
        )
        img.save("./img.png")
        for i in range(10):
            env.step(2)


if __name__ == "__main__":
    main()
