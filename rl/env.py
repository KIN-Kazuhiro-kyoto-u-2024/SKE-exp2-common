from collections import OrderedDict

import numpy as np
import math

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


# =====================================================================
# 報酬バリアント: (state_dict, config) -> (rew, best)
#   sweep.py から config.reward_variant で選ぶ。新しい報酬式を試したいときは
#   ここに関数を1つ書いて REWARD_VARIANTS に登録するだけ。
#   なお done（倒立失敗）時の -10 は _get_reward 側で短絡するのでここでは扱わない。
# =====================================================================
def _reward_default(sd, config):
    # 現状の挙動: 倒立を維持している限り毎ステップ +0.1
    return 0.1, 0

def _complex_reward(sd, config):
    
    # 報酬（rew）の設定
    # rew の与え方を色々変更してみる
    d = config.num_digitized
    n_pendulum_rad, n_pendulum_vel = sd["n_pendulum_rad"], sd["n_pendulum_vel"]
    n_best = (d - 1) / 2
    n_arm_rad, n_arm_vel = sd["n_arm_rad"], sd["n_arm_vel"]
    n_arm_best = (d - 1) / 2
    bonus = 0
    if n_pendulum_rad == n_best:
        bonus = 1
    else:
        bonus = 0
    rew = 10
    theta_diff = abs(n_arm_rad - n_arm_best) # 手前のうで
    alpha_diff = abs(n_pendulum_rad - n_best) # 先端のほう
    rew -= alpha_diff * 0.5
    rew -= abs(sd["arm_vel"]) * 0.1
    rew -= abs(sd["pendulum_vel"]) * 0.5
    rew -= abs(theta_diff) * 0.1
    return rew, 1 if bonus > 0 else 0

def _reward_alpha_only(sd, config):
    # 振り子角 alpha が中央（直立）に近いほど報酬を大きくする
    d = config.num_digitized
    n_best = (d - 1) / 2
    err = abs(sd["n_pendulum_rad"] - n_best) / n_best 
    # 0.0 〜 1.0。これを 1.0 から引いて報酬とする（err が小さいほど報酬が大きい）。
    # さらに err < 0.5 を best として返す（倒立状態の維持に成功しているかの指標）。
    return 1.0 - err, (1 if err < 0.5 else 0)

def _reward_alpha_nonlinear(sd, config):
    # 振り子角 alpha が中央（直立）に近いほど報酬を大きくする
    # ルートとったver
    d = config.num_digitized
    n_best = (d - 1) / 2
    err = abs(sd["n_pendulum_rad"] - n_best) / n_best
    # 0.0 〜 1.0。これを 1.0 から引いて報酬とする（err が小さいほど報酬が大きい）。
    # さらに err < 0.5 を best として返す（倒立状態の維持に成功しているかの指標）。
    return 1.0 - math.sqrt(err), (1 if err < 0.5 else 0)

def _reward_alpha_cos(sd, config):
    # 離散インデックス差の abs ではなく、実角 alpha（連続値）の cos を報酬にする。
    # 直立（alpha=0rad）で cos=1.0 が最大、傾くほど滑らかに減少する。
    alpha = sd["pendulum_rad"]
    rew = math.cos(alpha)
    # best: 直立範囲（±0.20π）の半分以内に入っていれば成功とみなす
    return rew, (1 if abs(alpha) < 0.10 * np.pi else 0)

def _reward_theta_decline(sd, config):
    # alpha_only をベースに
    # 手前のうで theta が中央に近いほど報酬を大きくする
    d = config.num_digitized
    n_best = (d - 1) / 2

    err_alpha = abs(sd["n_pendulum_rad"] - n_best) / n_best 
    err_theta = abs(sd["n_arm_rad"] - n_best) / n_best 
    return 1.0 - err_alpha - 0.1 * err_theta, (1 if err_alpha < 0.5 and err_theta < 0.5 else 0)


def _reward_theta_decline_2(sd, config):
    # alpha_only をベースに
    # 手前のうで theta が中央に近いほど報酬を大きくする
    d = config.num_digitized
    n_best = (d - 1) / 2

    err_alpha = abs(sd["n_pendulum_rad"] - n_best) / n_best 
    err_theta = abs(sd["n_arm_rad"] - n_best) / n_best 
    return 1.0 - err_alpha - 0.3 * err_theta, (1 if err_alpha < 0.5 and err_theta < 0.5 else 0)

def _reward_theta_decline_3(sd, config):
    # alpha_only をベースに
    # 手前のうで theta が中央に近いほど報酬を大きくする
    d = config.num_digitized
    n_best = (d - 1) / 2

    err_alpha = abs(sd["n_pendulum_rad"] - n_best) / n_best 
    err_theta = abs(sd["n_arm_rad"] - n_best) / n_best 
    return 1.0 - err_alpha - 0.6 * err_theta, (1 if err_alpha < 0.5 and err_theta < 0.5 else 0)

REWARD_VARIANTS = {
    "default": _reward_default,
    "alpha_only": _reward_alpha_only,
    "complex": _complex_reward,
    "theta_decline": _reward_theta_decline,
    "theta_decline_2": _reward_theta_decline_2,
    "theta_decline_3": _reward_theta_decline_3,
    "alpha_nonlinear": _reward_alpha_nonlinear,
    "alpha_cos": _reward_alpha_cos,
}


# =====================================================================
# 離散化バリアント: (low, high, d) -> 内側のビン境界（長さ d-1 の配列）
#   sweep.py から config.digitize_variant で選ぶ。連続値の刻み方を変えたいときは
#   ここに関数を1つ書いて DIGITIZE_VARIANTS に登録するだけ。
#   どれも中央 0 が重要点で対称な区間を想定しており、角度・速度の両方に使える。
# =====================================================================
def _edges_uniform(low, high, d):
    # 一様分割（従来挙動）。
    return np.linspace(low, high, d + 1)[1:-1]

def _edges_dense_center(low, high, d, power=2.0):
    # 中央（0付近=直立）を密にする非一様分割。power>1 で 0付近にビンが集まる。
    t = np.linspace(-1.0, 1.0, d + 1)
    warped = np.sign(t) * np.abs(t) ** power
    mid, half = (low + high) / 2.0, (high - low) / 2.0
    return (mid + warped * half)[1:-1]

DIGITIZE_VARIANTS = {
    "uniform": _edges_uniform,
    "dense_center": _edges_dense_center,
}


class Balance:

    def __init__(self, config):

        self._config = config
        d = config.num_digitized
        # 実機ランチャー（launch.py / launch_for_rl_policy.py）が参照する。
        # ゼロベクトルなので離散化境界は変わらない。
        self._arrange = np.zeros(d - 1)
        # 初期条件のランダム化範囲を config から渡す（学習できていない状態範囲のカバー用）。
        # 既定値は従来挙動（elbow ±0.1π, 他は 0）。sweep.py から SWEEP_INIT_* で振れる。
        self._env = balance(
            init_elbow_range=config.init_alpha_range,
            init_shoulder_range=config.init_theta_range,
            init_elbow_vel_range=config.init_alpha_vel_range,
            init_shoulder_vel_range=config.init_theta_vel_range,
        )
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

    def reset_to(self, alpha=0.0, theta=0.0, alpha_dot=0.0, theta_dot=0.0):
        """共通テスト用: 初期条件を厳密に指定してリセットする（ランダム化しない）。
        alpha=振り子(elbow)角, theta=腕(shoulder)角, *_dot は各角速度。
        全モデルに同一の初期条件セットを与えて公平に評価するために使う。"""
        self._env.reset()
        p = self._env.physics
        p.named.data.qpos["elbow"] = alpha
        p.named.data.qpos["shoulder"] = theta
        p.named.data.qvel["elbow"] = alpha_dot
        p.named.data.qvel["shoulder"] = theta_dot
        p.forward()  # 上書きした状態を派生量（向き等）に反映
        obs = OrderedDict()
        obs["orientations"] = p.orientations()
        obs["velocity"] = p.velocity()
        digitized_state, _, _ = self._digitized_state(obs)
        return digitized_state

    def _get_reward(self, state_dict, done, action):

        # 倒立状態の維持に失敗したときの報酬
        if done:
            return -10, 0

        # 報酬（rew）の設定は env.py 上部の REWARD_VARIANTS から選ぶ
        # （config.reward_variant で指定。sweep.py から SWEEP_REWARD で切替）
        return REWARD_VARIANTS[self._config.reward_variant](state_dict, self._config)

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
        # 離散化方法は env.py 上部の DIGITIZE_VARIANTS から選ぶ
        # （config.digitize_variant で指定。sweep.py から SWEEP_DIGITIZE で切替）
        edges = DIGITIZE_VARIANTS[self._config.digitize_variant]
        n_arm_rad = np.digitize(arm_rad, edges(self._arm_limit[0], self._arm_limit[1], d))
        n_arm_vel = np.digitize(arm_vel.clip(-8, 8), edges(-8, 8, d))
        n_pendulum_rad = np.digitize(
            pendulum_rad, edges(self._pendulum_limit[0], self._pendulum_limit[1], d)
        )
        n_pendulum_vel = np.digitize(pendulum_vel.clip(-8, 8), edges(-8, 8, d))

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
        # done を True としてエピソード終了。
        # done_reason に失敗原因を残す（train.py / sweep.py で原因別に集計する）:
        #   "theta" = θ(arm) 範囲外, "alpha" = α(pendulum) 範囲外, "both" = 両方同時, "" = 失敗なし
        arm_cond = n_arm_rad == 0 or n_arm_rad == d - 1
        pendulum_cond = n_pendulum_rad == 0 or n_pendulum_rad == d - 1
        if arm_cond and pendulum_cond:
            state_dict["done_reason"] = "both"
        elif arm_cond:
            state_dict["done_reason"] = "theta"
        elif pendulum_cond:
            state_dict["done_reason"] = "alpha"
        else:
            state_dict["done_reason"] = ""
        done = arm_cond or pendulum_cond
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
