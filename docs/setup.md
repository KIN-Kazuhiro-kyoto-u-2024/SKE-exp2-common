# 環境構築
システム工学実験　倒立振子　実行環境について

QUARC Python API を用いて，振子やモータを Python で制御している．
QUARC Python API の Documentation は https://docs.quanser.com/quarc/documentation/python/getting_started.html にある．

## Python の環境
Python の[公式サイト](https://www.python.org/)から，version 3.10.11 (Win64bit) をダウンロード&インストール．コマンドプロンプトから動かす．
インストールしたライブラリ
* numpy, opencv-python, matplotlib, scipy
* QUARC Python API に関するライブラリ
* control（極配置法で可制御性行列の計算などに用いる）

## 環境構築の手順について
下にまとめるが，基本的には [QUARC Python についての動画の4:44](https://www.youtube.com/watch?v=sRaoU9J6tP0)からの説明の通り．
1. Python の[公式サイト](https://www.python.org/)から，version 3.10.11 のインストーラをダウンロード．
2. インストーラを起動し，**Add Python 3.10 by PATH** にチェックを入れ，画面の指示に従ってインストール．
3. コマンドプロンプトで下記コマンドを実行
```bash
python -m pip install --upgrade pip
python -m pip install numpy opencv-python matplotlib scipy
cd 'C:/Program Files/Quanser/QUARC/python'                        （QUARC のインスト―ルディレクトリ直下にある「python」に移動）
python -m pip install --find-links . quanser_api-(以下略)       （QUARC Python API に関するライブラリをインストール，
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　以下略のところは dir コマンドで表示されるファイル名を見ればわかる）
python -m pip install control
```
## Python コードについて
デスクトップ上の QUARCpython というフォルダの中に一式入っている．コードは以下の３つのファイルに分割されている．
学生に編集してもらうのは基本的に launch.py だけになると思う：

* real/q2usb.py : QUARC Python API を呼び出し，振り子やモーターへとつながる信号を読み書きする．
* real/invpen.py : 倒立振子を表すクラス Invpen を実装．
	　　\theta（台座の回転角）, \alpha（振子の振れ角）やその時間微分（状態）を取得したり，トルクを入力したりするメソッドが実装されている．
　　　　	　　また，サンプル時間毎に実行するループの内容も，クラス Invpen の中にコーディングされている．
* launch.py : このプログラム中の Agent というクラスに制御則を実装する．
　　　　	　　そしてこの python コードをコマンドプロンプトから実行すると，倒立振子が起動し，Agent で実装した制御則が実機上で動作する．

## 正しくセットアップできているか確認
以下のコマンドを実行し、倒立振子が下に下がっている状態からスタートし、 以下のstartの文字が現れたら振子を真上に持ち上げる。
バランスタスクができれば成功！
```bash
python launch.py
```
```bash
***********
** start **
***********
```