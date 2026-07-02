"""共有メモリ1テーブルの並列Q学習ランナー（Hogwild 式）.

同じ Q テーブルを N 個のワーカープロセスで同時に更新して学習を高速化する。
sweep.py が「違う設定を並列比較」なのに対し、こちらは「同じ設定を並列で 1 つの
Q テーブルに集約」する。物理計算はこの問題では軽く（プロファイルで全体の ~8%）、
残りは Python オーバーヘッドなので、プロセスを増やすとほぼ台数分スループットが出る。

    python rl/train_parallel.py        # rl/ から実行（train.py と同じ作法）

学習に必要な「総」エピソード数（= EnvConfig.max_episode）は変わらない。それを
N ワーカーで分担して稼ぐので、実時間がおよそ 1/N になるイメージ（サンプル効率が
N 倍になるわけではない点に注意）。

------------------------------------------------------------------------------
共有しなければならない条件（Q テーブルの形・状態インデックスの意味・更新の整合性）:
  num_digitized / num_action / digitize_variant / reward_variant / alpha / gamma /
  epsilon / episode_length / max_episode。
  これらは EnvConfig（+ train.py と同じ SWEEP_* 環境変数）で全ワーカー共通になる。

ワーカーごとに変えてよい条件（env.reset の初期分布にしか効かない）:
  init_alpha (振り子 elbow 初期角), init_theta (腕 shoulder 初期角),
  init_alpha_vel (振り子初速度), init_theta_vel (腕初速度)。
  下の WORKER_INITS に「1 ワーカー = 1 辞書」で書く。ワーカー数 = len(WORKER_INITS)。
  各ワーカーに違う初速度・初期位置を割り当てると、1 つの共有 Q テーブルに広い状態
  範囲の経験が集まる（倒立点付近に偏りがちな探索を初期条件のばらけでカバー）。
------------------------------------------------------------------------------

メモ:
  * 更新はロックなし（Hogwild）。状態数が膨大（num_digitized**4）なので書き込み衝突は
    ほぼ起きず、Q 学習は多少の lost update に頑健。ロックを入れると速度が死ぬので入れない。
  * ログ・評価動画・モデル保存は worker 0 だけが担当（出力競合を避ける）。学習曲線の
    x 軸は全ワーカー合計の総エピソード数。worker 0 が見るのは自分のエピソードだけなので
    曲線は総経験のサンプルだが、傾向を見るには十分。
  * 再現性（決定論）は失われる。
"""

import dataclasses
import os
import pathlib
import shutil
import sys
import time
from collections import OrderedDict
from multiprocessing import Process, Value
from multiprocessing.shared_memory import SharedMemory

import numpy as np

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))

from env import make_env  # noqa: E402

# train.py の定義をそのまま再利用（EnvConfig / 環境変数上書き / Agent / config 保存）
from train import Agent, EnvConfig, _apply_env_overrides, _dump_config  # noqa: E402

from utils.logger import Logger  # noqa: E402


# ----------------------------------------------------------------------
# ここを編集してワーカーごとの初期条件を書く。ワーカー数 = len(WORKER_INITS)。
# 使えるキー: init_alpha / init_theta / init_alpha_vel / init_theta_vel
#   （未指定の項目は EnvConfig（+ SWEEP_INIT_* 環境変数）の既定値のまま）
# worker 0 がログ・評価・保存を担当するので、基準にしたい条件を先頭に置くとよい。
# ----------------------------------------------------------------------
WORKER_INITS = [
    {},                                                             # 00 既定（ログ担当）
    {"init_alpha": 0.05 * np.pi},                                   # 01 初期角 小
    {"init_alpha": 0.10 * np.pi},                                   # 02 初期角 中
    {"init_alpha_vel": 0.5},                                        # 03 振り子初速 小
    {"init_alpha_vel": 1.0},                                        # 04 振り子初速 中
    {"init_alpha_vel": 2.0},                                        # 05 振り子初速 大
    {"init_theta": 0.10 * np.pi},                                   # 06 腕初期角
    {"init_theta_vel": 0.5},                                        # 07 腕初速 小
    {"init_theta_vel": 1.0},                                        # 08 腕初速 大
    {"init_alpha_vel": 1.0, "init_theta_vel": 1.0},                 # 09 両初速
    {"init_alpha": 0.10 * np.pi, "init_alpha_vel": 1.0},            # 10 角＋初速
    {"init_alpha": 0.15 * np.pi, "init_alpha_vel": 2.0, "init_theta_vel": 1.0},  # 11 広範囲
]

# WORKER_INITS のキー -> EnvConfig のフィールド名
_INIT_FIELDS = {
    "init_alpha": "init_alpha_range",
    "init_theta": "init_theta_range",
    "init_alpha_vel": "init_alpha_vel_range",
    "init_theta_vel": "init_theta_vel_range",
}

_QDTYPE = np.float64  # models.Qtable の初期化（np.random.uniform）に合わせる
_FAIL_KEYS = ("theta", "alpha", "both", "timeout")


def _make_worker_config(base, init_override):
    """base 設定をコピーして、このワーカー用の初期条件だけ差し替える。"""
    cfg = dataclasses.replace(base)
    for key, value in init_override.items():
        setattr(cfg, _INIT_FIELDS[key], float(value))
    return cfg


def _attach_shared_q(shm_name, qshape):
    """名前付き共有メモリにアタッチして numpy ビューを返す（shm は閉じないよう保持）。"""
    shm = SharedMemory(name=shm_name)
    q = np.ndarray(qshape, dtype=_QDTYPE, buffer=shm.buf)
    return shm, q


def _select_action(shared_q, state, num_action, epsilon, explore=True):
    """共有 Q の行スナップショットから行動を選ぶ（Hogwild 競合対策）.

    models.get_action は「np.max → その後もう一度行を読んで ==max 比較」と行を 2 度
    読むため、その間に他プロセスが最大セルを書き換えると一致ゼロ＝空配列になり落ちる。
    ここでは行を 1 度だけ .copy() してから argmax するので一貫性が保たれる。
    更新は models.update_Qtable（単発 np.max のみ）をそのまま使うので問題ない。"""
    row = shared_q[state].copy()  # この決定の間だけ固定したスナップショット
    if explore and np.random.random() < epsilon:
        return int(np.random.randint(num_action))
    return int(np.random.choice(np.flatnonzero(row == row.max())))


def _run_episode(agent, env, cfg, shared_q):
    """1 エピソードを学習しながら実行し、共有 Q を更新。集計値を返す。"""
    state = env.reset()
    ep_reward = 0.0
    best_num = 0
    step = 0
    done = False
    state_dict = {"done_reason": "timeout", "best": 0}
    for step in range(cfg.episode_length):
        action = _select_action(shared_q, state, cfg.num_action, cfg.epsilon)
        next_state, reward, done, state_dict = env.step(action)
        agent.update_Qtable(state, action, reward, next_state)
        ep_reward += reward
        best_num += state_dict["best"]
        state = next_state
        if done:
            break
    reason = state_dict["done_reason"] if done else "timeout"
    return ep_reward, step, best_num, reason


def _evaluate_and_log(env, cfg, shared_q, logger, global_ep):
    """worker 0 用: グリーディ方策で 1 ロールアウトして動画・グラフを記録する。"""
    state = env.reset()
    eval_reward = 0.0
    best_num = 0
    img_seq, act_seq, rew_seq, alpha_seq, theta_seq = [], [], [], [], []
    for _ in range(cfg.video_length):
        img_seq.append(env._env.physics.render(height=480, width=640, camera_id=0))
        action = _select_action(shared_q, state, cfg.num_action, cfg.epsilon, explore=False)
        next_state, reward, done, sd = env.step(action)
        state = next_state
        act_seq.append(env._digitized_action[action])
        rew_seq.append(reward)
        alpha_seq.append(sd["pendulum_rad"])
        theta_seq.append(sd["arm_rad"])
        eval_reward += reward
        best_num += sd["best"]
        if done:
            break
    print(f"\n[eval @ep{global_ep}] reward: {eval_reward}, steps: {len(rew_seq)}\n", flush=True)
    logger.global_step = global_ep
    logger.log_video({"eval/video": np.array(img_seq)}, save=False)
    logger.add_scalars(
        OrderedDict([("eval/ep_reward", eval_reward), ("eval/ep_best_num", best_num)])
    )
    act_img = logger.plot2image("action", {"torque": np.array(act_seq)})
    rew_img = logger.plot2image("reward", {"reward": np.array(rew_seq)})
    state_img = logger.plot2image(
        "state", {"alpha": np.array(alpha_seq), "theta": np.array(theta_seq)}
    )
    logger.log_image(
        {"eval/action": act_img, "eval/reward": rew_img, "eval/state": state_img}
    )


def worker(wid, base, init_override, shm_name, qshape, max_episode, g_episode):
    """1 ワーカープロセス。自前の env を持ち、共有 Q を読み書きしながら学習する。
    wid==0 のときだけログ・評価・モデル保存を担当する。"""
    # ワーカーごとに乱数列をずらして探索を多様化（spawn 下で同一列になるのを防ぐ）
    np.random.seed((os.getpid() * 2654435761 + wid) & 0xFFFFFFFF)

    cfg = _make_worker_config(base, init_override)
    env = make_env(cfg)
    agent = Agent(cfg)
    shm, shared_q = _attach_shared_q(shm_name, qshape)
    # Agent 自前のランダム初期化を捨てて共有 Q に差し替える（models.py は触らない）
    agent._qtable._Qtable = shared_q

    is_logger = wid == 0
    if is_logger:
        logger = Logger(str(cfg.logdir))
        data_file = str(cfg.logdir) + "/data.csv"
        with open(data_file, "w") as df:
            print(
                "episode", "ave_ep_len", "ave_ep_rew", "qtable_err",
                "fail_theta", "fail_alpha", "fail_both", "fail_timeout",
                sep=",", file=df,
            )
        prev_q = shared_q.copy()
        ave_ep_len, ave_ep_rew = [], []
        fail_counts = {k: 0 for k in _FAIL_KEYS}
        last_scalar = last_model = last_video = 0

    try:
        while True:
            # 総エピソード数を atomically に1つ予約。上限に達したら終了。
            with g_episode.get_lock():
                if g_episode.value >= max_episode:
                    break
                g_episode.value += 1

            ep_reward, step, best_num, reason = _run_episode(
                agent, env, cfg, shared_q
            )

            if not is_logger:
                continue

            # ---- 以下 worker 0 のみ: 集計とログ ----
            ave_ep_len.append(step)
            ave_ep_rew.append(ep_reward)
            fail_counts[reason] += 1
            gnow = g_episode.value  # 全ワーカー合計の進捗（ログの x 軸）

            if gnow // base.should_log_scalar > last_scalar:
                last_scalar = gnow // base.should_log_scalar
                qtable_err = float(np.mean(np.abs(shared_q - prev_q)))
                prev_q = shared_q.copy()
                n = max(1, len(ave_ep_len))
                ave_len = sum(ave_ep_len) / n
                ave_rew = sum(ave_ep_rew) / n
                print(
                    f"[total {gnow}/{max_episode}] ave_len: {ave_len:.1f}, "
                    f"ave_rew: {ave_rew:.3f}",
                    flush=True,
                )
                logger.global_step = gnow
                logger.add_scalars(
                    OrderedDict(
                        [
                            ("train/ep_reward", ep_reward),
                            ("train/ep_length", step),
                            ("train/ep_best_num", best_num),
                            (f"train/ave_ep_len_{base.should_log_scalar}", ave_len),
                            ("train/qtable_error", qtable_err),
                            ("train/fail_theta", fail_counts["theta"]),
                            ("train/fail_alpha", fail_counts["alpha"]),
                            ("train/fail_both", fail_counts["both"]),
                            ("train/fail_timeout", fail_counts["timeout"]),
                        ]
                    )
                )
                with open(data_file, "a") as df:
                    print(
                        gnow, ave_len, ave_rew, qtable_err,
                        fail_counts["theta"], fail_counts["alpha"],
                        fail_counts["both"], fail_counts["timeout"],
                        sep=",", file=df,
                    )
                ave_ep_len, ave_ep_rew = [], []
                fail_counts = {k: 0 for k in _FAIL_KEYS}

            if gnow // base.should_log_model > last_model:
                last_model = gnow // base.should_log_model
                save_file = base.logdir.joinpath(f"qtable_{gnow}.npy")
                np.save(save_file, shared_q.copy())
                print(f"\nsave model {save_file}\n", flush=True)

            if gnow // base.should_log_video > last_video:
                last_video = gnow // base.should_log_video
                _evaluate_and_log(env, cfg, shared_q, logger, gnow)
    finally:
        shm.close()


def _copy_code(logdir):
    """train.py 同様、再現用にコードを logdir/codes へコピー（無ければ無視）。"""
    codes_dir = pathlib.Path(logdir) / "codes"
    codes_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.py", "train_parallel.py", "env.py", "eval.py", "models.py"):
        try:
            shutil.copy(name, codes_dir)
        except OSError:
            pass
    for name in ("../simulator/acrobot.py", "../simulator/acrobot.xml"):
        try:
            shutil.copy(name, codes_dir)
        except OSError:
            pass


def main():
    base = EnvConfig()
    base = _apply_env_overrides(base)  # train.py と同じ SWEEP_* 環境変数で上書き可

    os.makedirs(base.logdir, exist_ok=True)
    _dump_config(base)
    _copy_code(base.logdir)

    n_workers = len(WORKER_INITS)
    qshape = (base.state_size, base.num_action)
    print(f"log: {base.logdir}", flush=True)
    print(
        f"parallel Q-learning: {n_workers} workers sharing one "
        f"{qshape[0]}x{qshape[1]} Q-table",
        flush=True,
    )
    print(f"total episodes (across all workers): {base.max_episode}", flush=True)

    # 共有メモリ上に Q テーブルを確保して初期化（models.Qtable と同じ一様乱数）
    nbytes = int(np.prod(qshape)) * np.dtype(_QDTYPE).itemsize
    shm = SharedMemory(create=True, size=nbytes)
    shared_q = np.ndarray(qshape, dtype=_QDTYPE, buffer=shm.buf)
    shared_q[:] = np.random.uniform(-1.0, 1.0, size=qshape)
    if base.restore:
        print(f"restore: {base.restore_path}", flush=True)
        shared_q[:] = np.load(base.restore_path)

    g_episode = Value("L", 0)  # 全ワーカー合計の開始エピソード数

    procs = []
    t0 = time.time()
    for wid, override in enumerate(WORKER_INITS):
        p = Process(
            target=worker,
            args=(wid, base, override, shm.name, qshape, base.max_episode, g_episode),
            daemon=False,
        )
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\ninterrupted: terminating workers...", flush=True)
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()
    finally:
        final_file = base.logdir.joinpath("qtable_final.npy")
        np.save(final_file, shared_q.copy())
        print(f"\nsaved final model: {final_file}", flush=True)
        print(f"elapsed: {(time.time() - t0) / 60:.1f} min", flush=True)
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
