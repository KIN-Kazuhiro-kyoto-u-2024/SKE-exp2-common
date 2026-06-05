- 係数変更(k1)
    i3から係数を少し変更
    ```python
        def _get_reward(self, state_dict, done, action):

            # 倒立状態の維持に失敗したときの報酬
            if done:
                return -10, 0

            # 報酬（rew）の設定
            # rew の与え方を色々変更してみる
            d = self._config.num_digitized
            n_pendulum_rad, n_pendulum_vel = (
                state_dict["n_pendulum_rad"],
                state_dict["n_pendulum_vel"],
            )
            n_best = (d - 1) / 2
            n_arm_rad, n_arm_vel = state_dict["n_arm_rad"], state_dict["n_arm_vel"]
            n_arm_best = (d - 1) / 2

            # 目標値？とのズレ
            pendulum_rad_diff = abs(n_pendulum_rad - n_best)
            arm_rad_diff = abs(n_arm_rad - n_arm_best)
            pendulum_vel_diff = abs(n_pendulum_vel)
            arm_vel_diff = abs(n_arm_vel)

            epsilon = 10e-1  # 0.1 rad ~ 5 degrees
            bonus = (
                pendulum_rad_diff < epsilon
            )  # 振子の角度が中央に近ければ成功としてボーナス

            rew_list = (  # [要因, 重み]
                [-pendulum_rad_diff, 1],  # 振子の角度が中央からズレていないほど報酬
                [-arm_rad_diff, 3],  # アームの角度が中央からズレていないほど報酬
                # [arm_rad_diff > np.pi, -100],  # アーム角度が限界を超えて回っていると重い罰
                # ↑arm_rad_diffは離散化されてるのでこの比較は意味ないかも
                # しかも、arm_rad_diffが大きいとそもそもエピソードが終了するから意味ないかも
                [-pendulum_vel_diff, 0.1],  # 振子の速度がゆっくり（=安定？）なほど報酬
                [
                    -arm_vel_diff,  # アームの速度がゆっくり（=安定？）なほど報酬
                    0.1,  # あんまり遅いと落ちそうな時に追いつけない
                ],
            )  # 重みが自動調整できるといいのだが
            rew_alive = 5  # 生存報酬的な

            rew = rew_alive + sum(cause * weight for cause, weight in rew_list)
            return rew, 1 if bonus > 0 else 0
    ```
  - \06-04-20-00-18
