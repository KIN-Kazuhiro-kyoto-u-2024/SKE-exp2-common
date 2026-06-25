- デフォルト
   - 何もいじらず
   - \06-04-15-28-41\
- 位置のずれの絶対値に応じて減算
    ```python
        # 倒立状態の維持に失敗したときの報酬
        if done:
            return -10, 0

        # 報酬（rew）の設定
        # rew の与え方を色々変更してみる
        d = self._config.num_digitized
        n_pendulum_rad, n_pendulum_vel = state_dict["n_pendulum_rad"], state_dict["n_pendulum_vel"]
        n_best = (d - 1) / 2
        n_arm_rad, n_arm_vel = state_dict["n_arm_rad"], state_dict["n_arm_vel"]
        n_arm_best = (d - 1) / 2
        bonus = 0
        rew = 1.0 - 0.1 * (abs(n_pendulum_rad - n_best) + abs(n_arm_rad - n_arm_best))
        return rew, 1 if bonus > 0 else 0
    ```
  - \ 07-04-16-12-48\
- 角速度にもペナルティを設定
    ```python
        # 倒立状態の維持に失敗したときの報酬
        if done:
            return -10, 0

        # 報酬（rew）の設定
        # rew の与え方を色々変更してみる
        d = self._config.num_digitized
        n_pendulum_rad, n_pendulum_vel = state_dict["n_pendulum_rad"], state_dict["n_pendulum_vel"]
        n_best = (d - 1) / 2
        n_arm_rad, n_arm_vel = state_dict["n_arm_rad"], state_dict["n_arm_vel"]
        n_arm_best = (d - 1) / 2
        bonus = 0
        if n_pendulum_rad == n_best:
            bonus = 1
        else:
            bonus = 0
        rew = 10
        theta_diff = abs(n_arm_rad - n_arm_best) # 手前のうで
        alpha_diff = abs(n_pendulum_rad - n_best) # 先端のほう
        rew -= alpha_diff * 0.5
        rew -= abs(state_dict["arm_vel"]) * 0.1
        rew -= abs(state_dict["pendulum_vel"]) * 0.5
        rew -= abs(theta_diff) * 0.1
        return rew, 1 if bonus > 0 else
    ```
    - \06-04-17-19-38

- シンプル
  - alphaのみで報酬設定 ＆ 離散化を21段階に増やした
  
        num_digitized: int = 21


        # 倒立状態の維持に失敗したときの報酬
        if done:
            return -1.0, 0

        # 報酬（rew）の設定
        # rew の与え方を色々変更してみる
        d = self._config.num_digitized
        n_pendulum_rad, n_pendulum_vel = (
            state_dict["n_pendulum_rad"],
            state_dict["n_pendulum_vel"],
        )
        n_best = (d - 1) / 2    #中央
        n_arm_rad, n_arm_vel = state_dict["n_arm_rad"], state_dict["n_arm_vel"]
        n_arm_best = (d - 1) / 2
        bonus = 0
        alpha_err = abs(state_dict["n_pendulum_rad"] - n_best) / n_best  # 先端 alpha のずれ[0,1]
        theta_err = abs(state_dict["n_arm_rad"] - n_arm_best) / n_arm_best  # 手前のうで theta のずれ[0,1]
        alpha_vel = abs(state_dict["pendulum_vel"]) / 8.0  # 先端の角速度[0,~1]
        theta_vel = abs(state_dict["arm_vel"]) / 8.0  # うでの角速度[0,~1]

        #rew = 2.0 - (0.75 * alpha_err + 0.2 * theta_err + 0.15 * alpha_vel + 0.05 * theta_vel)
        rew = 2.0 - 1.0 * alpha_err
        rew = max(rew, 0.0)
        if alpha_err < (0.5 / n_best):
            bonus = 1
        return rew, 1 if bonus > 0 else 0


train_parallel logs\train\06-25-14-32-56

一番いいやつ 今のところ