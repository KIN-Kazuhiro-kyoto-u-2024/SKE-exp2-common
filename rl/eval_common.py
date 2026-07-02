"""学習方法が違うモデルを共通テストで公平に性能評価するランナー.

考え方:
  * 学習時の報酬（ave_ep_rew）は報酬式ごとにスケールも意味も違うので横並び比較できない。
    そこで **学習報酬に依存しないタスク指標**（生存ステップ数・成功率・平均|alpha|）で測る。
  * **同一の初期条件グリッド**を全モデルに与える（共通テストの本体）。env.reset_to() で
    初期条件を厳密指定するため、ランダム化に頼らず完全に再現・公平。
  * 観測→状態の離散化は **各モデルが学習時に使った方式**（config.json から復元）で行う。
    Qテーブルはその符号化に紐づくため、これを揃えないと評価が壊れる。

使い方:
    python rl/eval_common.py A.npy B.npy [C.npy ...]
  * 比較したい qtable の .npy パスを並べて渡すだけ。共通の初期条件グリッドで
    各モデルを評価し、成功率→平均生存ステップ順で並べて表示する。
  * .npy のほか、run ディレクトリ（中の最新 qtable_*.npy を使う）や
    ワイルドカード（"logs/train/**/qtable_*.npy" 等）も渡せる。
  * 各 .npy の符号化/報酬は同じ dir の config.json（無ければ dir 名トークン＋
    qtable 形状）から復元する。
  * 結果は標準出力の表に加えて eval_summary.csv（カレントディレクトリ）にも出力する。
"""

import glob as globmod
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))

from env import make_env, REWARD_VARIANTS, DIGITIZE_VARIANTS  # noqa: E402
from models import Qtable  # noqa: E402

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


def evaluate_qtable(qpath, name=None):
    """1 つの qtable(.npy) を共通バッテリで評価し、指標 dict を返す（評価不能なら None）。"""
    qpath = pathlib.Path(qpath)
    qtable = np.load(qpath)
    run_dir = qpath.parent
    cfg = _recover_config(run_dir, qtable)

    env = make_env(cfg)
    q = Qtable(cfg)
    q.load(qpath)
    if q._Qtable.shape != (cfg.state_size, cfg.num_action):
        # 形状不一致（config と qtable が食い違う）の場合は評価不能
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
        "name": name or run_dir.name,
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


def _write_csv(out_path, rows):
    with open(out_path, "w", encoding="utf-8") as f:
        print(",".join(COLS), file=f)
        for r in rows:
            print(",".join(str(r.get(c, "")) for c in COLS), file=f)


def _resolve_npy_paths(args):
    """引数（.npy / run dir / ワイルドカード）を .npy パスのリストに展開する。"""
    out = []
    for a in args:
        if any(c in a for c in "*?["):  # ワイルドカード
            matches = sorted(globmod.glob(a, recursive=True))
            if not matches:
                print(f"  (warn) マッチ無し: {a}", flush=True)
            out.extend(pathlib.Path(m) for m in matches)
            continue
        p = pathlib.Path(a)
        if p.is_dir():  # run ディレクトリ → 中の最新 qtable
            latest = _latest_qtable(p)
            if latest is None:
                print(f"  (warn) qtable_*.npy が無い dir: {a}", flush=True)
            else:
                out.append(latest)
            continue
        out.append(p)

    # .npy のみ＋実在のみに絞り、重複（同一実体）を順序維持で除去
    seen, uniq = set(), []
    for p in out:
        if p.suffix != ".npy" or not p.exists():
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _make_names(paths):
    """表示名を作る。基本は親 dir 名。重複する場合のみ qtable 名を付けて区別する。"""
    counts = {}
    for p in paths:
        counts[p.parent.name] = counts.get(p.parent.name, 0) + 1
    names = []
    for p in paths:
        base = p.parent.name
        names.append(f"{base}/{p.stem}" if counts[base] > 1 else base)
    return names


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(
            "使い方: python rl/eval_common.py A.npy B.npy ...\n"
            "  比較したい qtable の .npy パス（run dir / ワイルドカードも可）を渡してください。"
        )

    paths = _resolve_npy_paths(args)
    if not paths:
        sys.exit("評価できる .npy が見つかりませんでした。")
    names = _make_names(paths)

    print(f"{len(paths)} models, battery {len(BATTERY)} 初期条件 x horizon {HORIZON}\n",
          flush=True)

    rows = []
    for p, nm in zip(paths, names):
        print(f"  eval -> {nm}  ({p})", flush=True)
        res = evaluate_qtable(p, name=nm)
        if res is not None:
            rows.append(res)
        else:
            print(f"    (skip) 形状不一致で評価不能: {p}", flush=True)

    if not rows:
        sys.exit("評価可能なモデルがありませんでした。")

    # 成功率 → 平均生存ステップ の降順
    rows.sort(key=lambda r: (r["success_rate"], r["mean_steps"]), reverse=True)

    out = pathlib.Path("eval_summary.csv").resolve()
    _write_csv(out, rows)

    print("\n==== common-test result (success_rate desc) ====", flush=True)
    print(f"{'success':>8} {'mean_steps':>10} {'mean|a|':>8}  "
          f"{'dz':>12} {'reward':>12}  name", flush=True)
    for r in rows:
        print(f"{r['success_rate']:>8.3f} {r['mean_steps']:>10.1f} "
              f"{r['mean_abs_alpha']:>8.3f}  {r['digitize']:>12} "
              f"{r['reward']:>12}  {r['name']}", flush=True)
    print(f"\nsummary : {out}")


if __name__ == "__main__":
    main()
