import os
import pathlib
import shutil
import time
from dataclasses import dataclass

import numpy as np

from real.invpen import Invpen
from rl.env import Balance, make_env
from rl.models import Qtable
from utils.logger import Logger

@dataclass(frozen=True)
class EnvConfig:
    domain: str = "double_pendulum"
    task: str = "balance"
    repeat: int = 2
    num_digitized: int = 21  # 離散化の設定（学習時の設定値と合わせる）
    num_action: int = 7  # 離散化の設定（学習時の設定値と合わせる）
    state_size: int = num_digitized**4
    logdir: pathlib.Path = pathlib.Path().joinpath(
        "./logs/real-by-rl", str(time.strftime("%m-%d-%H-%M-%S"))
    )

    restore: bool = True
    # ここにシミュレータで学習した QTable のパス名を記載する
    restore_path: str = "log_sim_only/qtable_final (1)" \
    ".npy"

    # 以下のパラメータは実機実験では関係なし
    gamma: float = 0.99
    alpha: float = 0.5
    max_episode: int = int(10e4)
    episode_length: int = 500
    should_log_model: int = 10000
    should_log_scalar: int = 200
    should_log_video: int = 10000
    video_length: int = 200


class TrainedAgent:

    def __init__(self):

        config = EnvConfig()
        self.config = config
        # pretrained qtable
        self.qtable = Qtable(config)
        self.logger = Logger(config.logdir)
        print(config.restore_path)
        self.qtable.load(config.restore_path)
        self.env: Balance = make_env(config)  # Balance以外で動かすことを想定してない？
        self.start = False
        self.data = {"n_alpha": [np.nan], "n_theta": [np.nan]}

        # プログラムコードを log フォルダにコピー（後で見返す用）
        codes_dir = str(config.logdir) + "/codes"
        os.mkdir(codes_dir)
        shutil.copy("launch_for_rl_policy.py", codes_dir)

    def _digitized_obs(self, obs):
        """
        obs[0] theta
        obs[1] alpha
        obs[2] theta dot
        obs[3] alpha dot
        """

        # theta, thetadot, alpha, alphadot の離散化（学習時の設定と合わせる）
        d = self.config.num_digitized
        n_arm_rad = np.digitize(
            obs[0],
            np.linspace(self.env._arm_limit[0], self.env._arm_limit[1], d + 1)[1:-1],
        )
        n_arm_vel = np.digitize(obs[2].clip(-8, 8), np.linspace(-8, 8, d + 1)[1:-1])

        n_pen_rad = np.digitize(
            obs[1],
            np.linspace(
                self.env._pendulum_limit[0], self.env._pendulum_limit[1], d + 1
            )[1:-1]
            + self.env._arrange,
        )
        n_pen_vel = np.digitize(obs[3].clip(-8, 8), np.linspace(-8, 8, d + 1)[1:-1])

        state_dict = {}
        state_dict["n_arm_rad"] = n_arm_rad
        state_dict["n_pen_rad"] = n_pen_rad
        state_dict["digitized_state"] = (
            n_pen_rad + n_pen_vel * d + n_arm_vel * d**2 + n_arm_rad * d**3
        )
        arm_cond = n_arm_rad == 0 or n_arm_rad == d - 1
        pen_cond = n_pen_rad == 0 or n_pen_rad == d - 1
        if arm_cond or pen_cond:
            done = True
        else:
            done = False
        return state_dict["digitized_state"], done, state_dict

    def policy(self, obs):
        if self.start:
            digitized_state, done, state_dict = self._digitized_obs(obs)
            digitized_action = self.qtable.get_action(digitized_state, False)
            torque = self.env._digitized_action[digitized_action]
            self.data["n_alpha"].append(state_dict["n_pen_rad"])
            self.data["n_theta"].append(state_dict["n_arm_rad"])

            if done:
                self.start = False
                return 0
            if abs(obs[0]) > 0.8 * np.pi:
                self.start = False
                return 0
            return torque
        else:
            self.data["n_alpha"].append(np.nan)
            self.data["n_theta"].append(np.nan)
            if abs(obs[0]) < 0.8 * np.pi and abs(obs[1]) < np.pi / 50.0:
                self.start = True
            return 0

    def after_termination_func(self, data):

        # tensorboard.exe での出力用にデータを保存
        data.update({"n_alpha": self.data["n_alpha"], "n_theta": self.data["n_theta"]})
        data = {key: np.array(value) for key, value in data.items()}
        state_seq_img = self.logger.plot2image(
            "state", {"alpha": data["alpha"], "theta": data["theta"]}
        )
        n_state_seq_img = self.logger.plot2image(
            "n_state", {"n_alpha": data["n_alpha"], "n_theta": data["n_theta"]}
        )
        statedot_seq_img = self.logger.plot2image(
            "dot_state", {"alphadot": data["alphadot"], "thetadot": data["thetadot"]}
        )
        act_seq_img = self.logger.plot2image("action", {"torque": data["torque"]})
        time_seq_img = self.logger.plot2image("time", {"time": data["time"]})
        dt_seq_img = self.logger.plot2image("dt", {"dt": data["dt"]})
        alphaf_seq_img = self.logger.plot2image("alphaf", {"alphaf": data["alpha_f"]})
        img_dict = {
            "action": act_seq_img,
            "state/nodot": state_seq_img,
            "state/dot": statedot_seq_img,
            "state/alphaf": alphaf_seq_img,
            "state/n": n_state_seq_img,
            "time": time_seq_img,
            "dt": dt_seq_img,
        }
        self.logger.log_image(img_dict)

        # csv ファイルにも保存
        csv_file = str(self.config.logdir) + "/data.csv"
        with open(csv_file, "w") as df:
            print(*list(data.keys()), sep=",", file=df)
            data_values = np.array(list(data.values())).T
            for row in data_values:
                print(*list(row), sep=",", file=df)


def main():

    agent = TrainedAgent()
    my_invpen = Invpen(agent)

    # サンプル時間が 0.015s となっており，シミュレーションでのもの（0.02s）とずれているが，
    # これは意図したものであるので変更しないこと．
    # simulation_time に制御の実行時間[s]を指定する．
    my_invpen.run(sample_time=0.015, simulation_time=60.0, figure=True, logging=True)


if __name__ == "__main__":
    main()
