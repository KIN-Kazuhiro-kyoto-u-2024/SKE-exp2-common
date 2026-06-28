import numpy as np


def softmax(x):
    xmax = np.max(x)
    return np.exp(x - xmax) / np.sum(np.exp(x - xmax))


class Qtable:
    def __init__(self, config) -> None:
        self._Qtable = np.random.uniform(
            low=-1, high=1, size=(config.state_size, config.num_action)
        )
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.epsilon = getattr(config, "epsilon", 0.2)
        self.action_arr = np.arange(config.num_action)

    def load(self, path):
        self._Qtable = np.load(path)

    # 行動の選択
    def get_action(
        self, state, explore=True, global_step=None, method="epsilon-greedy"
    ) -> int:
        if not explore:
            max_action = np.where(self._Qtable[state] == np.max(self._Qtable[state]))[0]
            next_action = np.random.choice(max_action)
            return next_action
        elif method == "softmax":
            prob = softmax(self._Qtable[state])
            next_action = np.random.choice(self.action_arr, p=prob)
            return next_action
        elif method == "epsilon-greedy":
            eps = self.epsilon
            p = np.random.random()
            if p < eps:
                next_action = np.random.choice(self.action_arr)
            else:
                max_action = np.where(
                    self._Qtable[state] == np.max(self._Qtable[state])
                )[0]
                next_action = np.random.choice(max_action)
            return next_action
        elif method == "random":
            next_action = np.random.choice(self.action_arr)
            return next_action
        raise NotImplementedError

    def update_Qtable(self, state, action, reward, next_state):
        next_maxQ = np.max(self._Qtable[next_state])
        self._Qtable[state, action] = (1 - self.alpha) * self._Qtable[
            state, action
        ] + self.alpha * (reward + self.gamma * next_maxQ)
        debug = False
        if debug:
            print(f"Qtable: {self._Qtable}")
        return self._Qtable
