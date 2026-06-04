- デフォルト
   - 何もいじらず
   - \06-04-15-28-41\ 
- 位置のずれの絶対値に応じて減算

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

  - \ 07-04-16-12-48\