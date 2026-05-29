from asyncio.windows_events import INFINITE
from quanser.hardware import HIL, HILError, Clock

import time
import numpy as np

class Q2usb():

    r_encoder_channels = np.array([0, 1], dtype=np.int32)
    r_num_encoder_channels = len(r_encoder_channels)
    r_encoder_buffer = np.array([0, 0], dtype=np.int32)

    w_analog_channels = np.array([0], dtype=np.int32)
    w_num_analog_channels = len(w_analog_channels)
    w_analog_buffer = np.array([0], dtype=np.float64)

    def __init__(self, id='0'):

        self.q2usb = HIL('q2_usb', id)

        if self.q2usb.is_valid():
            counts = np.array([0, 0], dtype=np.int32)
            self.q2usb.set_encoder_counts(self.r_encoder_channels, self.r_num_encoder_channels, counts)

    def write_voltage(self, voltage):
        self.w_analog_buffer = -1.0 * np.array([voltage], dtype=np.float64)
        self.q2usb.write_analog(self.w_analog_channels, self.w_num_analog_channels, self.w_analog_buffer)

    def read_theta_and_alpha(self):
        self.q2usb.read_encoder(self.r_encoder_channels, self.r_num_encoder_channels, self.r_encoder_buffer)
        return np.pi*self.r_encoder_buffer/2048

    def terminate(self):
        self.write_voltage(0)
        self.q2usb.close()