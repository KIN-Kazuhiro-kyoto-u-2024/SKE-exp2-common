import dataclasses
import json
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
    num_digitized: int = 21  # 振れ角 theta および alpha に対する離散値の数
    num_action: int = 7  # アクション（入力トルク）に対する離散値の数
    state_size: int = num_digitized**4
    gamma: float = 0.99  # 割引率
    alpha: float = 0.5  # 学習率
    epsilon: float = 0.10  # epsilon-greedy の epsilon（models.py へ渡す）
    reward_variant: str = "theta_decline_3"  # env.py の REWARD_VARIANTS のキー
    digitize_variant: str = "uniform"  # env.py の DIGITIZE_VARIANTS のキー（離散化方式）
    # balance 開始時の初期条件ランダム化範囲（[-range, +range] 一様）。既定は従来挙動。
    init_alpha_range: float = 0.1 * np.pi  # 振り子(elbow)初期角度
    init_theta_range: float = 0.0  # 腕(shoulder)初期角度
    init_alpha_vel_range: float = 0.0  # 振り子初期角速度（初速度）
    init_theta_vel_range: float = 0.0  # 腕初期角速度（初速度）
    max_episode: int = int(10e5)  # 学習の総 episode 数
    episode_length: int = 2000  # 1 episode のタイムステップ数
    should_log_model: int = 1000  # 何 episode おきに QTable の値を保存するか
    should_log_scalar: int = 200  # 何 episode おきに学習のログを出力するか
    should_log_video: int = 1000  # 何 episode おきに QTable の評価をするか
    restore: bool = False  # 学習を再スタートする場合，これを True とする
    restore_path: str = ""  # QTable の初期値用データのパスを入力（再スタート時）
    video_length: int = 200  # QTable 評価時におけるタイムステップ数
    logdir: pathlib.Path = pathlib.Path().joinpath(
        "./logs/train", str(time.strftime("%m-%d-%H-%M-%S"))
    )


def _apply_env_overrides(config):
    """sweep.py から渡される環境変数で EnvConfig を上書きする。
    未設定なら何もしないので、単体実行時は従来挙動のまま。"""
    g = os.environ.get
    if g("SWEEP_ALPHA"):
        config.alpha = float(g("SWEEP_ALPHA"))
    if g("SWEEP_GAMMA"):
        config.gamma = float(g("SWEEP_GAMMA"))
    if g("SWEEP_EPS"):
        config.epsilon = float(g("SWEEP_EPS"))
    if g("SWEEP_REWARD"):
        config.reward_variant = g("SWEEP_REWARD")
    if g("SWEEP_DIGITIZE"):  # 離散化方式（env.py の DIGITIZE_VARIANTS のキー）
        config.digitize_variant = g("SWEEP_DIGITIZE")
    if g("SWEEP_INIT_ALPHA"):  # 振り子の初期角度ランダム化範囲
        config.init_alpha_range = float(g("SWEEP_INIT_ALPHA"))
    if g("SWEEP_INIT_THETA"):  # 腕の初期角度ランダム化範囲
        config.init_theta_range = float(g("SWEEP_INIT_THETA"))
    if g("SWEEP_INIT_ALPHA_VEL"):  # 振り子の初速度ランダム化範囲
        config.init_alpha_vel_range = float(g("SWEEP_INIT_ALPHA_VEL"))
    if g("SWEEP_INIT_THETA_VEL"):  # 腕の初速度ランダム化範囲
        config.init_theta_vel_range = float(g("SWEEP_INIT_THETA_VEL"))
    if g("SWEEP_NUM_DIGITIZED"):  # 状態の離散数（刻み幅）
        config.num_digitized = int(g("SWEEP_NUM_DIGITIZED"))
        config.state_size = config.num_digitized**4  # ← 必ず再計算
    if g("SWEEP_NUM_ACTION"):  # トルクの離散数（刻み幅）
        config.num_action = int(g("SWEEP_NUM_ACTION"))
    if g("SWEEP_MAX_EPISODE"):  # 動作確認用に総 episode 数を小さくできる
        config.max_episode = int(g("SWEEP_MAX_EPISODE"))
    if g("SWEEP_EPISODE_LENGTH"):  # 1 episode のタイムステップ数
        config.episode_length = int(g("SWEEP_EPISODE_LENGTH"))
    if g("SWEEP_LOGDIR"):
        config.logdir = pathlib.Path(g("SWEEP_LOGDIR"))
    return config


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


def _dump_config(config):
    """各 run の logdir に config.json を出力する。
    eval_common.py が離散化方式・num_digitized・num_action 等を確実に復元するため。"""
    d = dataclasses.asdict(config)
    d = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in d.items()}
    with open(pathlib.Path(config.logdir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def main():

    config = EnvConfig()
    config = _apply_env_overrides(config)
    env = make_env(config)
    os.makedirs(config.logdir, exist_ok=True)
    _dump_config(config)
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
    # should_log_scalar 窓ごとの失敗原因カウンタ（env.py の done_reason 別）
    fail_counts = {"theta": 0, "alpha": 0, "both": 0, "timeout": 0}
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
        print(
            "episode", "ave_ep_len", "ave_ep_rew", "qtable_err",
            "fail_theta", "fail_alpha", "fail_both", "fail_timeout",
            sep=",", file=df,
        )

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

        # 失敗原因を原因別にカウント（done なら done_reason、満了したなら timeout）
        if done:
            fail_counts[state_dict["done_reason"]] += 1
        else:
            fail_counts["timeout"] += 1

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
                        ("train/fail_theta", fail_counts["theta"]),
                        ("train/fail_alpha", fail_counts["alpha"]),
                        ("train/fail_both", fail_counts["both"]),
                        ("train/fail_timeout", fail_counts["timeout"]),
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
                    fail_counts["theta"],
                    fail_counts["alpha"],
                    fail_counts["both"],
                    fail_counts["timeout"],
                    sep=",",
                    file=df,
                )

            ave_ep_len = []
            ave_ep_rew = []
            fail_counts = {k: 0 for k in fail_counts}

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
