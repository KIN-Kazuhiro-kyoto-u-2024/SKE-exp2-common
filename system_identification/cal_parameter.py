from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# ここに同定用測定データのフォルダ名を記載する
DATA_FOR_Bp_DIR = './logs/real-for-sys-iden/only_pendulum/XX-XX-XX-XX-XX/'
DATA_FOR_Br_DIR = './logs/real-for-sys-iden/only_arm/XX-XX-XX-XX-XX/'
DATA_FOR_fl_DIR = './logs/real-for-sys-iden/static_fric/XX-XX-XX-XX-XX/'

@dataclass
class Param:
    g:float = 9.81
    Lr:float = 0.2159
    Jr:float = 9.9829e-4
    Br:float = 2.4e-3
    mr:float = 0.2570
    Lp:float = 0.3365
    Jp:float = 1.2e-3
    Bp:float = 2.4e-3
    mp:float = 0.1270
    etag:float = 0.69
    etam:float = 0.9
    Rm:float = 2.6
    Kg:float = 70.0
    kt:float = 7.68e-3
    km:float = 7.68e-3
    Am:float = Kg*kt/Rm
    Bm:float = Kg*km
P = Param()

# B_p の同定
def cal_Bp(data):

    time_seq = data['time']
    dt = data['dt']
    alpha = data['alpha']
    
    # alpha -> alpha_f の変換
    for i in range(len(alpha)):
        if alpha[i] > 0:
            alpha[i] = np.pi - alpha[i]
        else:
            alpha[i] = -np.pi - alpha[i]
    
    # 最大振幅までふれた時を初期時刻とする
    start_index = np.argmax(alpha)
    start_time = time_seq[start_index]
    alpha = alpha[start_index:]
    time_seq = time_seq[start_index:] - start_time
    y0 = alpha[0]

    # フィッティング関数の設定
    def fit_func(t, a, b):
        y = y0 * np.exp(-a*t)*np.cos(b*t)
        return y
    r0 = P.Jp + 0.25*P.mp*(P.Lp**2)
    a0 = P.Bp / (2*r0)
    b0 = np.sqrt(2*r0*P.mp*P.g*P.Lp - P.Bp**2)/(2*r0)
    init_params = [a0, b0]

    # フィッティングとその結果の出力
    print('init_param=', init_params)
    para_opt, _cov = curve_fit(fit_func, time_seq, alpha, init_params)
    print('opt param=', para_opt)
    r_opt = P.mp * P.g * P.Lp / (2 * (para_opt[0]**2 + para_opt[1]**2))
    print('Gamma_opt=', r_opt, ' Gamma=', r0)
    Bp_opt = 2*para_opt[0]*r_opt
    print('Bp_opt=', Bp_opt, ' Bp=', P.Bp)

    # プロット
    plt.plot(time_seq, alpha, label='raw data')
    fit_alpha = y0 * np.exp(-para_opt[0]*time_seq) * np.cos(para_opt[1]*time_seq)
    plt.plot(time_seq, fit_alpha, label='fit curve')
    no_fit_alpha = y0 * np.exp(-a0*time_seq) * np.cos(b0*time_seq)
    plt.plot(time_seq, no_fit_alpha, label='no fit curve')
    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

# B_r の同定
def cal_Br(data):

    thetadot_zero_index = np.where(data['thetadot'] == 0)[0]
    start, end = thetadot_zero_index[-1], -10
    time_seq = data['time'][start:end]
    time_seq = time_seq - time_seq[0]
    dt = data['dt'][start:end]
    theta = data['theta'][start:end]
    thetadot = data['thetadot'][start:end]
    voltage = data['voltage'][start:end]

    # フィッティング関数の設定
    def fit_func(x, a):
        t, v = x[0], x[1]
        y = ((P.Am * v) / (P.Am*P.Bm + a))*(1 - np.exp(-(t*(P.Am * P.Bm + a)) / P.Jr))
        return y
    a0 = P.Br
    init_params = [a0]

    # フィッティングとその結果の出力
    print('init_param=', init_params)
    x = np.array([time_seq, voltage])
    para_opt, cov = curve_fit(fit_func, x, thetadot, init_params)
    print('opt param=', para_opt)
    print('Br_opt=', para_opt[0], ' Br=', P.Bp)

    # プロット
    plt.plot(time_seq, thetadot, label='raw data')
    fit_br = ((P.Am * voltage) / (P.Am*P.Bm + para_opt[0]))*(1 - np.exp(-(time_seq*(P.Am * P.Bm + para_opt[0])) / P.Jr))
    plt.plot(time_seq, fit_br, label='fit curve')
    no_fit_br = ((P.Am * voltage) / (P.Am*P.Bm + P.Br))*(1 - np.exp(-(time_seq*(P.Am * P.Bm + P.Br)) / P.Jr))
    plt.plot(time_seq, no_fit_br, label='no fit curve')
    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

# 静止摩擦の同定
def cal_fric_loss(data):
    
    eps = 10e-5
    print(data['theta'])

    # アームが動き出した時刻のトルクを取得し出力
    start_to_move_index = np.where(np.abs(data['theta']) > eps)[0]
    start = start_to_move_index[0]
    friction_loss = data['torque'][start]
    print('friction_loss=', friction_loss)

def main():

    data_for_bp = np.load(DATA_FOR_Bp_DIR+'data.npy', allow_pickle=True).item()
    data_for_br = np.load(DATA_FOR_Br_DIR+'data.npy', allow_pickle=True).item()
    data_for_fl = np.load(DATA_FOR_fl_DIR+'data.npy', allow_pickle=True).item()
    
    cal_Bp(data_for_bp)
    cal_Br(data_for_br)
    cal_fric_loss(data_for_fl)

if __name__ == "__main__":
    main()