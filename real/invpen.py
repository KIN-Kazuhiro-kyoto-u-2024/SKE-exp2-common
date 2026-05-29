from real.quarc_hw import Q2usb

import time
import numpy as np
from matplotlib import pyplot as plt

class Invpen():

    #mechanical parameters of the inverted pendulum
    Lr = 0.2159
    Jr = 9.9829e-4
    Br = 2.4e-3
    mr = 0.2570
    Lp = 0.3365
    Jp = 1.2e-3
    Bp = 2.4e-3
    mp = 0.1270

    #parameters related to conversion from voltage to torque
    etag = 0.69
    etam = 0.9
    Rm = 2.6
    Kg = 70.0
    kt = 7.68e-3
    km = 7.68e-3
    Am = Kg*kt/Rm
    Bm = Kg*km

    #acceleration of gravity
    g = 9.81

    def __init__(self, controller, id='0'):

        self.io_device = Q2usb(id)
        self.agent = controller
        self.state = np.zeros(4)
        self.torque = 0.0
        self.data = {
            'theta': [0],
            'alpha': [np.pi],
            'thetadot': [0],
            'alphadot': [0],
            'alpha_f': [0],
            'torque': [0],
            'voltage': [0],
            'time': [0],
            'dt': [0]
        }
    
    def t2v(self):
        tau = self.torque
        td = self.state[2]
        return tau/Invpen.Am + td*Invpen.Bm
    
    def run(self, sample_time=0.05, simulation_time=1.0, figure=False, logging=False):
        timestamp = 0.0
        theta_prev = 0.0
        alpha_prev = np.pi
        start_prev = -sample_time
        # data_theta, data_alpha, data_thetadot, data_alphadot, \
        # data_alpha_f, data_torque, data_voltage, data_time = ([0], [np.pi], [0], [0], [0], [0], [0], [0])
        # data_dt = [0]

        print('')
        print('***********')
        print('** start **')
        print('***********')
        print('')

        start_time = time.time()
        def elapsed_time():
            return time.time() - start_time
        try:
            while timestamp < simulation_time:

                start = elapsed_time()

                #observe the current state
                theta, alpha_f = self.io_device.read_theta_and_alpha()

                alpha = alpha_f % (2*np.pi) - np.pi
                delta_t = start - start_prev
                thetadot = (theta - theta_prev) / delta_t
                alphadot = (alpha - alpha_prev) / delta_t

                #emergency stop
                if abs(theta) > np.pi :
                    break

                self.state[0] = theta
                self.state[1] = alpha
                self.state[2] = thetadot
                self.state[3] = alphadot

                #determine the input (torque) to the inverted pendulum,
                #according to the policy of your agent
                self.torque = self.agent.policy(self.state)

                #convert torque to voltage, and input it to the moter
                voltage = max(-5.0, min(5.0, self.t2v()))
                self.io_device.write_voltage(voltage)

                self.data['theta'].append(theta)
                self.data['alpha'].append(alpha)
                self.data['thetadot'].append(thetadot)
                self.data['alphadot'].append(alphadot)
                self.data['alpha_f'].append(alpha_f)
                self.data['torque'].append(self.torque)
                self.data['voltage'].append(voltage)
                self.data['time'].append(start)
                self.data['dt'].append(delta_t)
                theta_prev = theta
                alpha_prev = alpha
                start_prev = start

                end = elapsed_time()
                diff = sample_time - (end - start)
                while diff > 0.001:
                    end = elapsed_time()
                    diff = sample_time - (end - start)
                timestamp = elapsed_time()

        # except Exception as e:
        #     print(e)
        finally:
            self.io_device.terminate()
            self.agent.after_termination_func(self.data)
