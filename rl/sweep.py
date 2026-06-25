"""Q学習ハイパラの並列 sweep ランナー（外付け）.

下の ``configs`` に試したい設定（報酬式・離散化・epsilon・学習率・割引率）の
リストを書いて実行すると、各設定で ``train.py`` を subprocess として並列に回す。

    python rl/sweep.py            # リポジトリルートからでも rl/ からでも可

ポイント:
  * 各 run には設定が分かる名前のユニークな logdir を渡すので **出力先が衝突しない**。
  * さらに起動を ``STAGGER_SEC`` 秒ずつずらして、起動時の CPU/ディスク競合を避ける。
  * 同時実行数は ``MAX_PARALLEL`` 本まで。
  * 全 run 終了後、各 run の ``data.csv`` 末尾を集計して ``summary.csv`` に出力し、
    ``ave_ep_rew`` の高い順にコンソール表示する（良い設定を見つけやすくする）。

config の各キー（すべて任意。指定したものだけ上書きされ、未指定は train.py の既定値）:
  reward         : env.py の REWARD_VARIANTS のキー（例 "default", "center"）
  eps            : epsilon-greedy の epsilon
  alpha          : 学習率
  gamma          : 割引率
  num_digitized  : 状態の離散数（刻み幅）。指定すると state_size も再計算される
  num_action     : トルク（アクション）の離散数（刻み幅）
  max_episode    : 総 episode 数。動作確認で短く回したいときに指定
  episode_length : 1 episode のタイムステップ数（既定 200）
  video_length   : 評価動画のタイムステップ数（既定は episode_length に連動）
"""

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------
# ここを編集して sweep したい設定リストを書く
# ----------------------------------------------------------------------
configs = [
    # 離散化21段階、報酬式のみ変更
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
]

MAX_PARALLEL = 12  # 同時に走らせる最大本数
STAGGER_SEC = 5  # 連続する起動の間隔（秒）。出力衝突防止＆起動競合の緩和

# ----------------------------------------------------------------------
RL_DIR = pathlib.Path(__file__).resolve().parent  # train.py のある rl/

_launch_lock = threading.Lock()  # 起動を STAGGER_SEC 間隔に並べる
_proc_lock = threading.Lock()
_running_procs = set()


# config キー -> (train.py に渡す環境変数名, run 名に付ける接頭辞)
_KEYS = [
    ("reward", "SWEEP_REWARD", ""),
    ("eps", "SWEEP_EPS", "eps"),
    ("alpha", "SWEEP_ALPHA", "a"),
    ("gamma", "SWEEP_GAMMA", "g"),
    ("num_digitized", "SWEEP_NUM_DIGITIZED", "d"),
    ("num_action", "SWEEP_NUM_ACTION", "na"),
    ("max_episode", "SWEEP_MAX_EPISODE", "ep"),
    ("episode_length", "SWEEP_EPISODE_LENGTH", "len"),
    ("video_length", "SWEEP_VIDEO_LENGTH", "vlen"),
]


def _run_name(idx, cfg):
    parts = [f"run{idx:02d}"]
    for key, _envname, prefix in _KEYS:
        if key in cfg:
            parts.append(f"{prefix}{cfg[key]}")
    return "_".join(parts)


def _build_env(cfg, logdir):
    env = os.environ.copy()
    for key, envname, _prefix in _KEYS:
        if key in cfg:
            env[envname] = str(cfg[key])
    env["SWEEP_LOGDIR"] = str(logdir)
    env["PYTHONIOENCODING"] = "utf-8"  # console.log の文字化け防止
    return env


def run_one(idx, cfg, batch_dir):
    name = _run_name(idx, cfg)
    logdir = batch_dir / name
    logdir.mkdir(parents=True, exist_ok=True)
    env = _build_env(cfg, logdir)
    console_path = logdir / "console.log"

    # 起動だけを STAGGER_SEC 間隔に直列化（wait はロック外なので並列に走る）
    with _launch_lock:
        time.sleep(STAGGER_SEC)
        print(f"[{idx:02d}] start  -> {name}", flush=True)
        console = open(console_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "train.py"],
            cwd=str(RL_DIR),
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
        )
        with _proc_lock:
            _running_procs.add(proc)

    ret = proc.wait()
    console.close()
    with _proc_lock:
        _running_procs.discard(proc)
    print(f"[{idx:02d}] done   -> {name} (ret={ret})", flush=True)
    return idx, name, ret


def _write_summary(batch_dir, results):
    # data.csv の header: episode,ave_ep_len,ave_ep_rew,qtable_err
    rows = []
    for idx, name, ret in results:
        data_csv = batch_dir / name / "data.csv"
        ep = ave_len = ave_rew = None
        if data_csv.exists():
            lines = data_csv.read_text().strip().splitlines()
            if len(lines) >= 2:
                cols = lines[-1].split(",")
                if len(cols) >= 3:
                    ep, ave_len, ave_rew = cols[0], float(cols[1]), float(cols[2])
        rows.append((idx, name, ret, ep, ave_len, ave_rew))

    # ave_ep_rew 降順（未取得は末尾）
    rows.sort(key=lambda r: (r[5] is not None, r[5] if r[5] is not None else 0), reverse=True)

    out = batch_dir / "summary.csv"
    with open(out, "w") as f:
        print("name,ret,episode,ave_ep_len,ave_ep_rew", file=f)
        for _idx, name, ret, ep, ave_len, ave_rew in rows:
            print(name, ret, ep, ave_len, ave_rew, sep=",", file=f)

    print("\n==== summary (ave_ep_rew desc) ====", flush=True)
    print(f"{'ave_ep_rew':>12}  {'ave_ep_len':>10}  name", flush=True)
    for _idx, name, ret, ep, ave_len, ave_rew in rows:
        flag = "" if ret == 0 else f"  [FAILED ret={ret}]"
        print(f"{str(ave_rew):>12}  {str(ave_len):>10}  {name}{flag}", flush=True)
    print(f"\nsummary : {out}")
    print(f"compare : tensorboard --logdir {batch_dir}")


def main():
    ts = time.strftime("%m-%d-%H-%M-%S")
    batch_dir = RL_DIR / "logs" / "train" / f"sweep_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "configs.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False))

    print(f"batch dir : {batch_dir}")
    print(f"{len(configs)} runs, max {MAX_PARALLEL} parallel, stagger {STAGGER_SEC}s\n", flush=True)

    results = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
            futures = [ex.submit(run_one, i, c, batch_dir) for i, c in enumerate(configs)]
            for fut in as_completed(futures):
                results.append(fut.result())
    except KeyboardInterrupt:
        print("\ninterrupted: terminating child processes...", flush=True)
        with _proc_lock:
            for p in list(_running_procs):
                p.terminate()
        raise

    _write_summary(batch_dir, results)


if __name__ == "__main__":
    main()
