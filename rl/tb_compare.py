"""複数の TensorBoard ログディレクトリを比較表示するランチャー（外付け・読み取り専用）.

複数の run のディレクトリを指定すると、1つの TensorBoard に名前付きで読み込み、
``train/*`` のスカラーだけをフィルタした状態でブラウザを開く。

    python rl/tb_compare.py <dir1> <dir2> [...]

例（``rl/logs/train/`` 配下はディレクトリ名だけで指定可）:

    python rl/tb_compare.py 06-05-16-38-34 sweep_06-11-17-04-25
    python rl/tb_compare.py rl/logs/train/06-05-16-38-34 C:\\full\\path\\to\\logdir

ポイント:
  * TensorBoard はイベントファイルを**読むだけ**なので、実行中の train.py には干渉しない。
  * デフォルトポートは 6014。既存の ``tensorboard --logdir ...``（6006）と同時に使える。
  * sweep バッチディレクトリ（sweep_*）を丸ごと渡すと、配下の各 run が自動展開される。
  * ``--filter`` でタグの正規表現フィルタを変更できる（既定 ``^train``）。
    ブラウザの URL フィルタが効かない場合は、画面左上の「Filter tags」欄に
    train と入力すれば同じ表示になる。
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser

RL_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = RL_DIR / "logs" / "train"


def _resolve(arg):
    """引数を「そのまま（カレント基準）→ rl/logs/train 基準」の順で探す。"""
    for cand in (pathlib.Path(arg), DEFAULT_LOG_ROOT / arg):
        if cand.is_dir():
            return cand.resolve()
    return None


def _make_names(paths):
    """logdir_spec 用の run 名。末尾名が重複したら親ディレクトリ名を前置する。"""
    leaves = [p.name for p in paths]
    names = []
    for p in paths:
        name = p.name if leaves.count(p.name) == 1 else f"{p.parent.name}_{p.name}"
        # , と : は logdir_spec の区切り文字なので使えない
        names.append(name.replace(",", "-").replace(":", "-"))
    seen = {}
    uniq = []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        uniq.append(n)
    return uniq


def main():
    parser = argparse.ArgumentParser(
        description="複数のログディレクトリを1つの TensorBoard で比較表示する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("logdirs", nargs="+", help="比較したいログディレクトリ（複数可）")
    parser.add_argument("--port", type=int, default=6014, help="TensorBoard のポート（既定 6014）")
    parser.add_argument("--filter", default="^train", help="タグフィルタの正規表現（既定 ^train）")
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    paths = []
    for arg in args.logdirs:
        path = _resolve(arg)
        if path is None:
            print(f"error: ディレクトリが見つかりません: {arg}")
            print(f"       （カレントからの相対 / 絶対 / {DEFAULT_LOG_ROOT} 配下の名前で指定）")
            sys.exit(1)
        paths.append(path)

    names = _make_names(paths)
    spec = ",".join(f"{n}:{p}" for n, p in zip(names, paths))

    tb = shutil.which("tensorboard")
    cmd = [tb] if tb else [sys.executable, "-m", "tensorboard.main"]
    cmd += [
        "--logdir_spec", spec,
        "--port", str(args.port),
        "--reload_multifile", "true",  # 学習中の run もライブ更新で追従
    ]

    url = f"http://localhost:{args.port}/#timeseries&tagFilter={urllib.parse.quote(args.filter)}"
    print("runs:")
    for n, p in zip(names, paths):
        print(f"  {n}  <-  {p}")
    print(f"\nurl : {url}")
    print("（タグフィルタが効かない場合は画面左上の Filter tags 欄に train と入力）\n", flush=True)

    proc = subprocess.Popen(cmd)
    try:
        if not args.no_browser:
            time.sleep(4)
            if proc.poll() is not None:  # ポート使用中などで起動に失敗
                sys.exit(proc.returncode)
            webbrowser.open(url)
        sys.exit(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
