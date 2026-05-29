from dataclasses import dataclass
import time
import pathlib

import numpy as np
import control as ctrl

from rl.models import Qtable
from utils.logger import Logger
from real.invpen import Invpen
from rl.env import make_env

@dataclass
class EnvConfig:
    domain: str = "double_pendulum"
    task: str = "balance"
    repeat: int = 2
    num_digitized: int = 8
    num_action: int = 4
    state_size: int = num_digitized**4
    gamma: float = 0.99
    alpha: float = 0.5
    max_episode: int = int(10e7)
    episode_length: int = 500
    should_log_model: int = 10000
    should_log_scalar: int = 200
    should_log_video: int = 10000
    restore: bool = True
    restore_path: str = "logs/10-13-06-26-05/qtable_100000.npy"
    video_length: int = 200
    logdir: str = pathlib.Path().joinpath("./logs/real", str(time.strftime("%m-%d-%H-%M-%S")))


class Agent():

    def __init__(self, eps):

        # only use log
        config = EnvConfig()
        self.config = config
        self.qtable = Qtable(config)
        self.logger = Logger(config.logdir)
        print(config.restore_path)
        # self.qtable.load(config.restore_path)
        self.env = make_env(config)
        self.start = False
        self.data = {'n_alpha': [], 'n_theta': []}

        ################################
        ### policy by pole placement ###
        ################################
        
        Lr = Invpen.Lr
        Jr = Invpen.Jr
        Br = Invpen.Br
        mr = Invpen.mr
        Lp = Invpen.Lp
        Jp = Invpen.Jp
        Bp = Invpen.Bp
        mp = Invpen.mp
        g = Invpen.g

        #des_poles = [-3+4i, -3-4i, -30, -40]
        des_coeffs = [1, 76, 1645, 8950, 30000]

        c4, c3, c2, c1 = des_coeffs[1:5]

        detK = (mp*Lr**2+Jr)*(0.25*mp*Lp**2+Jp) - 0.25*(mp*Lr*Lp)**2
        A = np.array([[0,0,1,0], [0,0,0,1], \
                      [0, 0.25*(mp**2)*(Lp**2)*Lr*g/detK, -0.25*(mp*Lp**2+4.0*Jp)*Br/detK, -0.5*Lr*Lp*mp*Bp/detK],
                      [0, 0.5*mp*Lp*g*(mp*Lr**2+Jr)/detK, -0.5*Lr*Br*Lp*mp/detK, -(mp*Lr**2+Jr)*Bp/detK]])
        print('A = ')
        print(A)

        B = np.array([[0], [0], [0.25*(mp*Lp**2+4.0*Jp)/detK], [0.5*mp*Lr*Lp/detK]])
        print('B = ')
        print(B)

        T = ctrl.ctrb(A, B)
        print('T = ')
        print(T)

        eig_A = np.linalg.eigvals(A)
        poly_A = np.poly(eig_A)
        a4, a3, a2, a1 = poly_A[1:5]

        tildeA = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[-a1,-a2,-a3,-a4]])
        tildeB = np.array([[0], [0], [0], [1]])
        tildeT = ctrl.ctrb(tildeA, tildeB)
        W = np.matmul(T, np.linalg.inv(tildeT))

        print('tildeA = ')
        print(tildeA)
        print('W = ')
        print(W)

        tildeK = np.array([c1-a1, c2-a2, c3-a3, c4-a4])

        self.K = np.linalg.solve(W.T, tildeK)

        print('K = ')
        print(self.K)

        self.eps = eps
        print('eps/PI = ')
        print(eps/np.pi)

    def _digitized_obs(self, obs):
        '''
        obs[0] theata
        obs[1] alpha
        obs[2] theta dot
        obs[3] alpha dot
        '''
        d = self.config.num_digitized
        n_arm_rad = np.digitize(obs[0],
                                np.linspace(self.env._arm_limit[0], self.env._arm_limit[1], d +1)[1:-1])
        n_arm_vel = np.digitize(obs[2].clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])

        n_pen_rad = np.digitize(obs[1],
                                np.linspace(self.env._pendulum_limit[0], self.env._pendulum_limit[1], d+1)[1:-1] + self.env._arrange)
        n_pen_vel = np.digitize(obs[3].clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])

        state_dict = {}
        state_dict["n_arm_rad"] = n_arm_rad
        state_dict["n_pen_rad"] = n_pen_rad
        state_dict["digitized_state"] = n_pen_rad + n_pen_vel*d + n_arm_vel*d**2 + n_arm_rad*d**3
        arm_cond = n_arm_rad == 0 or n_arm_rad == d - 1
        pen_cond = n_pen_rad == 0 or n_pen_rad == d -1
        if arm_cond or pen_cond:
            done = True
        else:
            done = False
        return state_dict["digitized_state"], done, state_dict

    def policy(self, state):
        if self.start:
            if abs(state[1]) < self.eps:
                action = -np.dot(self.K, state)
                digitized_state, done, state_dict = self._digitized_obs(state)
                torque = self.qtable.get_action(digitized_state, False, 0, method=None)
                self.data['n_alpha'].append(state_dict['n_pen_rad'])
                self.data['n_theta'].append(state_dict['n_arm_rad'])
                return action
            else:
                return 0.00
        else:
            if abs(state[0]) < 0.8*np.pi and abs(state[1]) < self.eps:
                self.start =True
            return 0

    def after_termination_func(self, data):
        data.update({'n_alpha': self.data['n_alpha'], 'n_theta': self.data['n_theta']})
        data = {key: np.array(value) for key, value in data.items()}
        state_seq_img = self.logger.plot2image('state', {'alpha': data['alpha'],
                                                    'theta': data['theta']})
        statedot_seq_img = self.logger.plot2image('dot_state', {'alphadot': data['alphadot'],
                                                                'thetadot': data['thetadot']})
        n_state_seq_img = self.logger.plot2image('n_state', {'n_alpha': data['n_alpha'],
                                                        'n_theta': data['n_theta']})
        act_seq_img = self.logger.plot2image('torque', {'torque': data['torque']})
        time_seq_img = self.logger.plot2image('time', {'time': data['time']})
        dt_seq_img = self.logger.plot2image('dt', {'dt': data['dt']})
        alphaf_seq_img = self.logger.plot2image('alphaf', {'alphaf': data['alpha_f']})
        img_dict = {'action': act_seq_img,
                    'state/nodot': state_seq_img,
                    'state/dot': statedot_seq_img,
                    'state/alphaf': alphaf_seq_img,
                    'state/n': n_state_seq_img,
                    'time': time_seq_img,
                    'dt': dt_seq_img}
        self.logger.log_image(img_dict)

def main():
    my_agent = Agent(eps=np.pi/15.0)
    my_invpen = Invpen(my_agent)
    my_invpen.run(sample_time=0.010, simulation_time=10.0, figure=True, logging=True)

if __name__ == "__main__":
    main()