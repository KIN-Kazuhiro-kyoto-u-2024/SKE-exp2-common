import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))

from collections import OrderedDict

from env import make_env
from models import Qtable

from utils.logger import Logger


@dataclass
class EnvConfig:
    domain: str = "double_pendulum"
    task: str = "balance"
    repeat: int = 2
    num_digitized: int = 8  # 振れ角 theta および alpha に対する離散値の数
    num_action: int = 4  # アクション（入力トルク）に対する離散値の数
    state_size: int = num_digitized**4
    gamma: float = 0.99  # 割引率
    alpha: float = 0.5  # 学習率
    max_episode: int = int(10e4)  # 学習の総 episode 数
    episode_length: int = 200  # 1 episode のタイムステップ数
    should_log_model: int = 1000  # 何 episode おきに QTable の値を保存するか
    should_log_scalar: int = 200  # 何 episode おきに学習のログを出力するか
    should_log_video: int = 1000  # 何 episode おきに QTable の評価をするか
    restore: bool = False  # 学習を再スタートする場合，これを True とする
    restore_path: str = ""  # QTable の初期値用データのパスを入力（再スタート時）
    video_length: int = 200  # QTable 評価時におけるタイムステップ数
    logdir: str = pathlib.Path().joinpath(
        "./logs/train", str(time.strftime("%m-%d-%H-%M-%S"))
    )


class Agent:
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
    print(f"log: {config.logdir}")
    logger = Logger(config.logdir)
    agent = Agent(config)

    # ランダムな行動を使ってQtableを初期化
    for i in range(100):
        state = env.reset()
        for step in range(400):
            action = agent.get_action(state, explore=True, method="random")
            next_state, reward, done, _ = env.step(action)
            qtable = agent.update_Qtable(state, action, reward, next_state)
            state = next_state
            if done:
                break

    # 再スタートする場合，Qtableの値をロード
    if config.restore:
        print(config.restore_path)
        qtable = np.load(config.restore_path)
        agent._qtable._Qtable = qtable

    ave_ep_len = []
    ave_ep_rew = []
    prev_q = agent._qtable._Qtable.copy()
    total_time = 0

    # プログラムコードを log フォルダにコピー（後で見返す用）
    codes_dir = str(config.logdir) + "/codes"
    os.mkdir(codes_dir)
    shutil.copy("train.py", codes_dir)
    shutil.copy("env.py", codes_dir)
    shutil.copy("eval.py", codes_dir)
    shutil.copy("models.py", codes_dir)
    shutil.copy("../simulator/acrobot.py", codes_dir)
    shutil.copy("../simulator/acrobot.xml", codes_dir)

    # 学習データ保存用の csv ファイルを作成
    data_file = str(config.logdir) + "/data.csv"
    with open(data_file, "w") as df:
        print("episode", "ave_ep_len", "ave_ep_rew", "qtable_err", sep=",", file=df)

    # main training loop
    for episode in range(config.max_episode + 1):

        # １エピソードの実行
        start_time = time.time()
        state = env.reset()
        episode_reward = 0
        best_num = 0
        for step in range(config.episode_length):
            action = agent.get_action(
                state, logger.global_step, method="epsilon-greedy"
            )
            next_state, reward, done, state_dict = env.step(action)
            qtable = agent.update_Qtable(state, action, reward, next_state)
            episode_reward += reward
            best_num += state_dict["best"]
            state = next_state
            if done:
                break
        ave_ep_len.append(step)
        ave_ep_rew.append(episode_reward)

        # スカラーをログに保存
        if episode % config.should_log_scalar == 0:
            print(
                f"episode: {episode}, episode_reward: {episode_reward}, episode_steps: {step+1}"
            )
            qtable_err = np.mean(np.abs((agent._qtable._Qtable - prev_q)))
            prev_q = agent._qtable._Qtable.copy()
            logger.add_scalars(
                OrderedDict(
                    [
                        ("train/ep_reward", episode_reward),
                        ("train/ep_length", step),
                        ("train/ep_best_num", best_num),
                        (
                            f"train/ave_ep_len_{config.should_log_scalar}",
                            sum(ave_ep_len) / len(ave_ep_len),
                        ),
                        ("train/qtable_error", qtable_err),
                        ("time/total_minute", total_time / 60),
                    ]
                )
            )

            # csv ファイルにも保存
            with open(data_file, "a") as df:
                print(
                    episode,
                    sum(ave_ep_len) / len(ave_ep_len),
                    sum(ave_ep_rew) / len(ave_ep_rew),
                    qtable_err,
                    sep=",",
                    file=df,
                )

            ave_ep_len = []
            ave_ep_rew = []

        # qtableを保存
        if episode % config.should_log_model == 0 and episode != 0:
            save_file = config.logdir.joinpath(f"qtable_{episode}.npy")
            np.save(save_file, qtable)
            print(f"\nsave model {save_file}\n")

        # 評価と動画の保存
        if episode % config.should_log_video == 0 and episode != 0:
            state = env.reset()
            eval_reward = 0
            best_num = 0
            img_seq = []
            act_seq = []
            img_seq = []
            rew_seq = []
            alpha_seq = []
            theta_seq = []
            for _ in range(config.video_length):
                img_seq.append(
                    env._env.physics.render(height=480, width=640, camera_id=0)
                )
                action = agent.get_action(state, explore=False)
                next_state, reward, done, state_dict = env.step(action)
                state = next_state
                act_seq.append(env._digitized_action[action])
                rew_seq.append(reward)
                alpha_seq.append(state_dict["pendulum_rad"])
                theta_seq.append(state_dict["arm_rad"])
                eval_reward += reward
                best_num += state_dict["best"]
                if done:
                    break
            print(
                f"\nevaluate episode reward: {eval_reward}, episode step: {len(rew_seq)}\n"
            )
            logger.log_video({"eval/video": np.array(img_seq)}, save=False)
            logger.add_scalars(
                OrderedDict(
                    [("eval/ep_reward", eval_reward), ("eval/ep_best_num", best_num)]
                )
            )
            act_seq_img = logger.plot2image("action", {"torque": np.array(act_seq)})
            rew_seq_img = logger.plot2image("reward", {"reward": np.array(rew_seq)})
            state_seq_img = logger.plot2image(
                "state", {"alpha": np.array(alpha_seq), "theta": np.array(theta_seq)}
            )
            img_dict = {
                "eval/action": act_seq_img,
                "eval/reward": rew_seq_img,
                "eval/state": state_seq_img,
            }
            logger.log_image(img_dict)
        total_time += time.time() - start_time
        start_time = time.time
        logger.step()


if __name__ == "__main__":
    main()
