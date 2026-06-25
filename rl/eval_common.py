"""学習方法が違うモデルを共通テストで公平に性能評価するランナー.

考え方:
  * 学習時の報酬（ave_ep_rew）は報酬式ごとにスケールも意味も違うので横並び比較できない。
    そこで **学習報酬に依存しないタスク指標**（生存ステップ数・成功率・平均|alpha|）で測る。
  * **同一の初期条件グリッド**を全モデルに与える（共通テストの本体）。env.reset_to() で
    初期条件を厳密指定するため、ランダム化に頼らず完全に再現・公平。
  * 観測→状態の離散化は **各モデルが学習時に使った方式**（config.json から復元）で行う。
    Qテーブルはその符号化に紐づくため、これを揃えないと評価が壊れる。

使い方:
    python rl/eval_common.py [SWEEP_BATCH_DIR | all]
  * 引数省略時は logs/train/sweep_* の最新バッチを自動評価する。
  * "all" を渡すと logs/train 配下の sweep_* バッチを全部評価する。
  各 run（qtable_*.npy を持つサブディレクトリ）を走査し、共通グリッドで評価して
  バッチ直下に eval_summary.csv（成功率→平均生存ステップ順）を出力する。
  all モードでは横断比較用に logs/train/eval_summary_all.csv も併せて出力する。
"""

import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))

from env import make_env, REWARD_VARIANTS, DIGITIZE_VARIANTS  # noqa: E402
from models import Qtable  # noqa: E402

RL_DIR = pathlib.Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# 共通テストバッテリ（初期条件グリッド）。ここを編集して試す範囲を変える。
#   theta(腕)は 0 固定、alpha(振り子)角と alpha 初速度を格子状に網羅する。
#   どの状態領域で倒立を維持できるかを公平に比較できる。
# ----------------------------------------------------------------------
ALPHA0_GRID = np.linspace(-0.15 * np.pi, 0.15 * np.pi, 7)  # 振り子初期角 [rad]
ALPHADOT0_GRID = np.linspace(-2.0, 2.0, 5)                 # 振り子初速度 [rad/s]
THETA0 = 0.0       # 腕初期角
THETADOT0 = 0.0    # 腕初速度
HORIZON = 1000     # 1 エピソードの最大ステップ数（これだけ倒立維持できたら成功）

BATTERY = [
    (float(a0), THETA0, float(ad0), THETADOT0)
    for a0 in ALPHA0_GRID
    for ad0 in ALPHADOT0_GRID
]


def _latest_qtable(run_dir):
    qs = sorted(run_dir.glob("qtable_*.npy"),
                key=lambda p: int(p.stem.split("_")[1]))
    return qs[-1] if qs else None


def _recover_config(run_dir, qtable):
    """run の config.json（無ければ run 名 + qtable 形状）から評価用 config を復元する。"""
    cfg = SimpleNamespace(
        domain="double_pendulum", task="balance", repeat=2,
        num_digitized=None, num_action=None,
        reward_variant="default", digitize_variant="uniform",
        gamma=0.99, alpha=0.5, epsilon=0.2,
        init_alpha_range=0.0, init_theta_range=0.0,
        init_alpha_vel_range=0.0, init_theta_vel_range=0.0,
    )
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        for k, v in data.items():
            setattr(cfg, k, v)
    else:
        # フォールバック: run 名トークンから digitize を拾う（"dz<value>"）
        for tok in run_dir.name.split("_"):
            if tok.startswith("dz"):
                cfg.digitize_variant = tok[2:]

    # num_digitized / num_action は qtable 形状からも確定できる: shape=(d^4, num_action)
    state_size, num_action = qtable.shape
    if cfg.num_digitized is None:
        cfg.num_digitized = int(round(state_size ** 0.25))
    if cfg.num_action is None:
        cfg.num_action = int(num_action)
    cfg.state_size = cfg.num_digitized ** 4

    # 復元値が壊れていないか軽く検証
    if cfg.digitize_variant not in DIGITIZE_VARIANTS:
        cfg.digitize_variant = "uniform"
    if cfg.reward_variant not in REWARD_VARIANTS:
        cfg.reward_variant = "default"
    return cfg


def evaluate_run(run_dir):
    """1 run を共通バッテリで評価し、指標 dict を返す（評価不能なら None）。"""
    qpath = _latest_qtable(run_dir)
    if qpath is None:
        return None
    qtable = np.load(qpath)
    cfg = _recover_config(run_dir, qtable)

    env = make_env(cfg)
    q = Qtable(cfg)
    q.load(qpath)
    if q._Qtable.shape != (cfg.state_size, cfg.num_action):
        # 形状不一致（config と qtable が食い違う）の場合は qtable 側に合わせる
        return None

    total_steps = 0
    successes = 0
    abs_alpha_sum = 0.0
    abs_alpha_n = 0
    fails = {"theta": 0, "alpha": 0, "both": 0}

    for (a0, th0, ad0, thd0) in BATTERY:
        state = env.reset_to(alpha=a0, theta=th0, alpha_dot=ad0, theta_dot=thd0)
        survived = 0
        sd = None
        for _step in range(HORIZON):
            action = q.get_action(state, explore=False)
            state, _reward, done, sd = env.step(action)
            abs_alpha_sum += abs(sd["pendulum_rad"])
            abs_alpha_n += 1
            survived += 1
            if done:
                break
        total_steps += survived
        if survived >= HORIZON:  # ホライズン満了まで倒立維持＝成功
            successes += 1
        elif sd is not None and sd.get("done_reason") in fails:
            fails[sd["done_reason"]] += 1

    n = len(BATTERY)
    return {
        "name": run_dir.name,
        "d": cfg.num_digitized,
        "na": cfg.num_action,
        "digitize": cfg.digitize_variant,
        "reward": cfg.reward_variant,
        "n_ic": n,
        "mean_steps": total_steps / n,
        "success_rate": successes / n,
        "mean_abs_alpha": (abs_alpha_sum / abs_alpha_n) if abs_alpha_n else float("nan"),
        "fail_theta": fails["theta"],
        "fail_alpha": fails["alpha"],
        "fail_both": fails["both"],
    }


COLS = ["name", "d", "na", "digitize", "reward", "n_ic",
        "mean_steps", "success_rate", "mean_abs_alpha",
        "fail_theta", "fail_alpha", "fail_both"]


def _pick_batch_dirs():
    """評価対象のバッチ dir 一覧を返す（引数なし=最新1件 / "all"=全件 / dir 指定）。"""
    train_dir = RL_DIR / "logs" / "train"
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            sweeps = sorted(train_dir.glob("sweep_*"))
            if not sweeps:
                sys.exit("評価対象が見つかりません（logs/train/sweep_* が無い）。")
            return sweeps
        return [pathlib.Path(sys.argv[1])]
    sweeps = sorted(train_dir.glob("sweep_*"))
    if not sweeps:
        sys.exit("評価対象が見つかりません。sweep バッチ dir を引数で指定してください。")
    return [sweeps[-1]]


def _write_csv(out_path, rows):
    with open(out_path, "w", encoding="utf-8") as f:
        print(",".join(COLS), file=f)
        for r in rows:
            print(",".join(str(r.get(c, "")) for c in COLS), file=f)


def evaluate_batch(batch_dir):
    """1 バッチを評価し eval_summary.csv を書き出して、成功率降順の rows を返す。"""
    run_dirs = sorted(p for p in batch_dir.iterdir()
                      if p.is_dir() and any(p.glob("qtable_*.npy")))
    if not run_dirs:
        print(f"  (skip) qtable を持つ run が無い: {batch_dir.name}", flush=True)
        return []

    print(f"batch dir : {batch_dir}")
    print(f"{len(run_dirs)} runs, battery {len(BATTERY)} 初期条件 x horizon {HORIZON}\n",
          flush=True)

    rows = []
    for rd in run_dirs:
        print(f"  eval -> {rd.name}", flush=True)
        res = evaluate_run(rd)
        if res is not None:
            rows.append(res)

    # 成功率 → 平均生存ステップ の降順
    rows.sort(key=lambda r: (r["success_rate"], r["mean_steps"]), reverse=True)

    out = batch_dir / "eval_summary.csv"
    _write_csv(out, rows)

    print("\n==== common-test result (success_rate desc) ====", flush=True)
    print(f"{'success':>8} {'mean_steps':>10} {'mean|a|':>8}  "
          f"{'dz':>12} {'reward':>12}  name", flush=True)
    for r in rows:
        print(f"{r['success_rate']:>8.3f} {r['mean_steps']:>10.1f} "
              f"{r['mean_abs_alpha']:>8.3f}  {r['digitize']:>12} "
              f"{r['reward']:>12}  {r['name']}", flush=True)
    print(f"\nsummary : {out}")
    return rows


def main():
    batch_dirs = _pick_batch_dirs()
    all_rows = []
    for batch_dir in batch_dirs:
        print(f"\n========== {batch_dir.name} ==========", flush=True)
        for r in evaluate_batch(batch_dir):
            all_rows.append({**r, "batch": batch_dir.name})

    # 複数バッチを評価した場合は横断比較用 CSV をまとめて出力する
    if len(batch_dirs) > 1:
        all_rows.sort(key=lambda r: (r["success_rate"], r["mean_steps"]), reverse=True)
        out_all = RL_DIR / "logs" / "train" / "eval_summary_all.csv"
        with open(out_all, "w", encoding="utf-8") as f:
            print("batch," + ",".join(COLS), file=f)
            for r in all_rows:
                print(r["batch"] + "," + ",".join(str(r.get(c, "")) for c in COLS),
                      file=f)
        print(f"\n==== all batches summary ({len(all_rows)} runs) ====")
        print(f"combined summary : {out_all}")


if __name__ == "__main__":
    main()
