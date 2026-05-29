# Q-learning
## Code Structure
`rl/train.py` - main function for training and evaluate agent

`rl/models.py` - Qtable

`utils/logger.py` - Logger

## Setup
必要なライブラリ
tensorboard dm-control torch torchvision torchaudio opencv-python moviepy

## For training
```bash
python train.py
```

## tensorboard
学習状況を見る
```bash
tensorboard --logdir ./logs
```

## Args
train.pyのEnvConfigの値を適宜変更する。

注意点: shouderの角度の分割数は0.5*num_digitized
```python
class EnvConfig:
    domain: str = "double_pendulum"     # only double_pendulum
    task: str = "swingup"               # swingup or balance
    num_digitized: int = 16             # １つの値を何分割するか
    num_action: int = 2                 # actionの分割数
    state_size: int = num_digitized**4  # 状態の総数
    gamma: float = 0.99                 # 割引率
    alpha: float = 0.5                  # qtableの更新率
    max_episode: int = int(10e7)        # 学習するエピソードの総数
    episode_length: int = 400           # １エピソードの長さ
    should_log_model: int = 10000       # 何ステップごとにqtableを保存するか
    should_log_scalar: int = 100        # 何ステップごとにrewardをlogに加えるか
    should_log_video: int = 10000       # 何ステップごとに現在のqtableを評価し動画にするか
    restore: bool = False               # 学習済みのqtableを使用するか
    restore_file: str = "Qtable.npy"    # 学習済みのqtableのパス
    video_length: int = 400             # 評価するときのエピソードの長さ
    logdir: str = "./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "/" #logを残すディレクトリ
```