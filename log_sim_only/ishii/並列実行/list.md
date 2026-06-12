# 第1回
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