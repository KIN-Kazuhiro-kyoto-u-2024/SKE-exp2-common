# 第1回

    python .\rl\tb_compare.py rl\logs\train\sweep_06-06-18-37-45\run00_default_eps0.2_a0.5_g0.99_d8_na4 rl\logs\train\sweep_06-06-18-37-45\run01_alpha_only_eps0.2_a0.5_g0.99_d8_na4 rl\logs\train\sweep_06-06-18-37-45\run02_center_eps0.2_a0.5_g0.99_d12_na4 rl\logs\train\sweep_06-06-18-37-45\run03_default_eps0.2_a0.5_g0.99_d24_na4 rl\logs\train\sweep_06-06-18-37-45\run04_alpha_only_eps0.2_a0.5_g0.75_d8_na4 rl\logs\train\sweep_06-06-18-37-45\run05_alpha_only_eps0.2_a0.5_g0.5_d8_na4 rl\logs\train\sweep_06-06-18-37-45\run06_alpha_only_eps0.2_a0.75_g0.99_d8_na4 rl\logs\train\sweep_06-06-18-37-45\run07_alpha_only_eps0.2_a0.25_g0.99_d8_na4

    configs = [
        # 全部デフォルト
        {"reward": "default", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 8, "num_action": 4},
        # alphaのみで報酬設定、ほかはそのまま
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 8, "num_action": 4},
        # alphaのみ、状態の離散化幅を大きく
        {"reward": "center", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 12, "num_action": 4},
        {"reward": "default", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 24, "num_action": 4},
        # alphaのみ、gammaを小さく
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.75, "num_digitized": 8, "num_action": 4},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.50, "num_digitized": 8, "num_action": 4},
        # alphaのみ、alphaを変える
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.75, "gamma": 0.99, "num_digitized": 8, "num_action": 4},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.25, "gamma": 0.99, "num_digitized": 8, "num_action": 4},
    ]

# 第2回

     python .\rl\tb_compare.py rl\logs\train\sweep_06-11-17-04-25\run00_default_eps0.2_a0.5_g0.99_d8_na4_ep1000000 rl\logs\train\sweep_06-11-17-04-25\run01_alpha_only_eps0.2_a0.5_g0.99_d8_na4_ep1000000 rl\logs\train\sweep_06-11-17-04-25\run02_complex_eps0.2_a0.5_g0.99_d8_na4_ep1000000 rl\logs\train\sweep_06-11-17-04-25\run03_alpha_only_eps0.2_a0.5_g0.99_d24_na4_ep1000000 rl\logs\train\sweep_06-11-17-04-25\run04_alpha_only_eps0.2_a0.5_g0.99_d48_na4_ep1000000 rl\logs\train\sweep_06-11-17-04-25\run05_alpha_only_eps0.2_a0.5_g0.75_d24_na4_ep1000000 rl\logs\train\sweep_06-11-17-04-25\run06_alpha_only_eps0.2_a0.5_g0.5_d24_na4_ep1000000 rl\logs\train\sweep_06-11-17-04-25\run07_alpha_only_eps0.2_a0.5_g0.25_d24_na4_ep1000000 

    configs = [
        # 全部デフォルト、報酬式のみ変更
        {"reward": "default", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 8, "num_action": 4, "max_episode": int(10e5)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 8, "num_action": 4, "max_episode": int(10e5)},
        {"reward": "complex", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 8, "num_action": 4, "max_episode": int(10e5)},
        # 離散化幅を大きく、alpha_only
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 24, "num_action": 4, "max_episode": int(10e5)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 48, "num_action": 4, "max_episode": int(10e5)},

        # alpha_only でgammaを小さく (d24固定)
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.75, "num_digitized": 24, "num_action": 4, "max_episode": int(10e5)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.50, "num_digitized": 24, "num_action": 4, "max_episode": int(10e5)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.25, "num_digitized": 24, "num_action": 4, "max_episode": int(10e5)},
    ]

# 第3回

    python .\rl\tb_compare.py rl\logs\train\sweep_06-12-16-09-02\run00_alpha_only_eps0.2_a0.5_g0.99_d21_na4_ep100000 rl\logs\train\sweep_06-12-16-09-02\run01_alpha_nonlinear_eps0.2_a0.5_g0.99_d21_na4_ep100000 rl\logs\train\sweep_06-12-16-09-02\run02_complex_eps0.2_a0.5_g0.99_d21_na4_ep100000 rl\logs\train\sweep_06-12-16-09-02\run03_theta_decline_eps0.2_a0.5_g0.99_d21_na4_ep100000 rl\logs\train\sweep_06-12-16-09-02\run04_theta_decline_2_eps0.2_a0.5_g0.99_d21_na4_ep100000 rl\logs\train\sweep_06-12-16-09-02\run05_theta_decline_3_eps0.2_a0.5_g0.99_d21_na4_ep100000

    configs = [
        # 離散化21段階、報酬式のみ変更
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 4, "max_episode": int(10e4)},
        {"reward": "alpha_nonlinear", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 4, "max_episode": int(10e4)},
        {"reward": "complex", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 4, "max_episode": int(10e4)},
        {"reward": "theta_decline", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 4, "max_episode": int(10e4)},
        {"reward": "theta_decline_2", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 4, "max_episode": int(10e4)},
        {"reward": "theta_decline_3", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 4, "max_episode": int(10e4)},
    ]
# 第4回 中断
# 第5回

    python .\rl\tb_compare.py rl\logs\train\sweep_06-12-17-21-25\run00_alpha_only_eps0.2_a0.5_g0.99_d21_na4_ep100000 rl\logs\train\sweep_06-12-17-21-25\run01_alpha_only_eps0.2_a0.5_g0.99_d21_na5_ep100000 rl\logs\train\sweep_06-12-17-21-25\run02_alpha_only_eps0.2_a0.5_g0.99_d21_na6_ep100000 rl\logs\train\sweep_06-12-17-21-25\run03_alpha_only_eps0.2_a0.5_g0.99_d21_na7_ep100000 rl\logs\train\sweep_06-12-17-21-25\run04_alpha_only_eps0.2_a0.5_g0.99_d21_na12_ep100000 rl\logs\train\sweep_06-12-17-21-25\run05_alpha_only_eps0.2_a0.5_g0.99_d21_na24_ep100000 rl\logs\train\sweep_06-12-17-21-25\run06_alpha_only_eps0.2_a0.5_g0.99_d21_na36_ep100000 rl\logs\train\sweep_06-12-17-21-25\run07_alpha_only_eps0.2_a0.5_g0.99_d21_na48_ep100000

    configs = [
        # 離散化21段階、報酬式のみ変更
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 4, "max_episode": int(10e4)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 5, "max_episode": int(10e4)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 6, "max_episode": int(10e4)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 12, "max_episode": int(10e4)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 24, "max_episode": int(10e4)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 36, "max_episode": int(10e4)},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 48, "max_episode": int(10e4)},
    ]

# 第6回
    configs = [
        # 離散化21段階、報酬式のみ変更
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 200},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.5, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 200},
        {"reward": "alpha_only", "eps": 0.75, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 200},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.75, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 200},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.25, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 200},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 35, "num_action": 7, "max_episode": int(10e4), "episode_length": 200},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 13, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.5, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.75, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.5, "alpha": 0.5, "gamma": 0.99, "num_digitized": 35, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.5, "alpha": 0.5, "gamma": 0.50, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        
    ]

# 第7回
    configs = [
        # 離散化21段階、報酬式のみ変更
        {"reward": "alpha_only", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "complex", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline_2", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline_3", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_nonlinear", "eps": 0.2, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},

        
        {"reward": "alpha_only", "eps": 0.15, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        
    ]

# 第8回
    configs = [
        # 離散化21段階、報酬式のみ変更
        {"reward": "alpha_only", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.05, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "alpha_only", "eps": 0.01, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},

        {"reward": "theta_decline", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline", "eps": 0.05, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline", "eps": 0.01, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},

        {"reward": "theta_decline_2", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline_2", "eps": 0.05, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline_2", "eps": 0.01, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        
        {"reward": "theta_decline_3", "eps": 0.10, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline_3", "eps": 0.05, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
        {"reward": "theta_decline_3", "eps": 0.01, "alpha": 0.5, "gamma": 0.99, "num_digitized": 21, "num_action": 7, "max_episode": int(10e4), "episode_length": 2000},
    ]

# 第9回

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