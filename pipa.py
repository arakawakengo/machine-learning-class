import numpy as np
from control import matlab
from matplotlib import pylab as plt
plt.rcParams["font.family"] = "MS Gothic"
import csv
import pandas as pd

def read_csv_data(path):
    csv_input1 = pd.read_csv(filepath_or_buffer=path, header=1, usecols=[0, 1])
    x = csv_input1.values
    csv_input2 = pd.read_csv(filepath_or_buffer=path, header=1, usecols=[0, 2])
    y = csv_input2.values
    return x, y

def Cs_a(R1=20000,R2=300000):
    a0 = R2/R1
    return matlab.tf([a0], [1])

def Cs_b(R1=4700000,R2=1000000,R3=2000,R4=7500,C1=1e-5,C2=1e-6):
    a1 = C2 * R4
    a0 = 1
    b1 = C1 * R4 * R2 / R1
    b0 = R2 * R4 /(R3 * R1)
    return matlab.tf([b1, b0], [a1, a0])

def Ps_spe(TM=0.064516129,TE=0.002409639,K=1.25):
    a2 = TM * TE
    a1 = TM + TE
    a0 = 1
    b0 = K
    return matlab.tf([b0], [a2, a1, a0])

def Ps_pos(TM=0.064516129,TE=0.002409639,K=208.3333333):
    a3 = TM * TE
    a2 = TM + TE
    a1 = 1
    a0 = 0
    b0 = K
    return matlab.tf([b0], [a3, a2, a1, a0])


def v_in_a_spe():
    Cs = Cs_a()
    Ps = Ps_spe()
    ch1, ch2 = read_csv_data(r"C:\Users\zzzke\Downloads\速度制御\a_in_こっちがホント.csv")
    sys = matlab.feedback(Cs*Ps, 1)
    t = np.linspace(0, 0.1, 1000)
    yout, T = matlab.step(sys*0.84, t)
    plt.plot(ch1.T[0], ch1.T[1], color = "black", label = "v_ref")
    plt.plot(ch2.T[0], ch2.T[1], color = "green", label = "v_in")
    plt.xlabel("time/s")
    plt.ylabel("voltage/V")
    plt.grid()
    t1= np.linspace(-0.01, 0.4, 1000)
    plt.plot(t1, step(t1)*0.84, color="b", linestyle="--", label = "ステップ信号")
    plt.plot(T, yout, color = "red", label = "シミュレーション結果")
    plt.xlim(-0.01, 0.1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    plt.show()

def v_out_a_spe():
    Cs = Cs_a()
    Ps = Ps_spe()
    ch1, ch2 = read_csv_data(r"C:\Users\zzzke\Downloads\速度制御\a_out_1.csv")
    sys = matlab.feedback(Cs, Ps)
    t = np.linspace(0, 0.1, 1000)
    yout, T = matlab.step(sys*0.88, t)
    plt.plot(ch1.T[0], ch1.T[1], color = "black", label = "v_ref")
    plt.plot(ch2.T[0], ch2.T[1], color = "green", label = "v_out")
    plt.xlabel("time/s")
    plt.ylabel("voltage/V")
    plt.grid()
    t1= np.linspace(-0.01, 0.4, 1000)
    plt.plot(t1, step(t1)*0.88, color="b", linestyle="--", label = "ステップ信号")
    plt.plot(T, yout, color = "red", label = "シミュレーション結果")
    plt.xlim(-0.01, 0.1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    plt.show()

def v_in_b_spe():
    Cs = Cs_b()
    Ps = Ps_spe()
    ch1, ch2 = read_csv_data(r"C:\Users\zzzke\Downloads\速度制御\b_in_2_こっち.csv")
    sys = matlab.feedback(Cs*Ps, 1)
    t = np.linspace(0, 0.11, 1000)
    yout, T = matlab.step(sys*9, t)
    plt.plot(ch1.T[0], ch1.T[1], color = "black", label = "v_ref")
    plt.plot(ch2.T[0], ch2.T[1], color = "green", label = "v_in")
    plt.xlabel("time/s")
    plt.ylabel("voltage/V")
    plt.grid()
    t1= np.linspace(-0.01, 0.4, 1000)
    plt.plot(t1, step(t1)*9, color="b", linestyle="--", label = "ステップ信号")
    plt.plot(T, yout, color = "red", label = "シミュレーション結果")
    plt.xlim(-0.01, 0.11)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    plt.show()

def v_out_b_spe():
    Cs = Cs_b()
    Ps = Ps_spe()
    ch1, ch2 = read_csv_data(r"C:\Users\zzzke\Downloads\速度制御\b_out_1.csv")
    sys = matlab.feedback(Cs, Ps)
    t = np.linspace(0, 0.11, 1000)
    yout, T = matlab.step(sys*2.9, t)
    plt.plot(ch1.T[0], ch1.T[1], color = "black", label = "v_ref")
    plt.plot(ch2.T[0], ch2.T[1], color = "green", label = "v_out")
    plt.xlabel("time/s")
    plt.ylabel("voltage/V")
    plt.grid()
    t1= np.linspace(-0.01, 0.4, 1000)
    plt.plot(t1, step(t1)*2.9, color="b", linestyle="--", label = "ステップ信号")
    plt.plot(T, yout, color = "red", label = "シミュレーション結果")
    plt.xlim(-0.01, 0.11)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    plt.show()

def v_in_b_pos():
    Cs = Cs_b()
    Ps = Ps_pos()
    ch1, ch2 = read_csv_data(r"C:\Users\zzzke\Downloads\位置制御\b_in.csv")
    sys = matlab.feedback(Cs*Ps, 1)
    t = np.linspace(0, 0.4, 1000)
    yout, T = matlab.step(sys*5.4, t)
    plt.plot(ch1.T[0], ch1.T[1], color = "black", label = "v_ref")
    plt.plot(ch2.T[0], ch2.T[1], color = "green", label = "v_in")
    plt.xlabel("time/s")
    plt.ylabel("voltage/V")
    plt.grid()
    t1= np.linspace(-0.01, 0.4, 1000)
    plt.plot(t1, step(t1)*5.4, color="b", linestyle="--", label = "ステップ信号")
    plt.plot(T, yout, color = "red", label = "シミュレーション結果")
    plt.xlim(-0.01, 0.4)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    plt.show()

def v_out_b_pos():
    Cs = Cs_b()
    Ps = Ps_pos()
    ch1, ch2 = read_csv_data(r"C:\Users\zzzke\Downloads\位置制御\b_out.csv")
    sys = matlab.feedback(Cs, Ps)
    t = np.linspace(0, 0.4, 1000)
    yout, T = matlab.step(sys*5.4, t)
    plt.plot(ch1.T[0], ch1.T[1], color = "black", label = "v_ref")
    plt.plot(ch2.T[0], ch2.T[1], color = "green", label = "v_out")
    plt.xlabel("time/s")
    plt.ylabel("voltage/V")
    plt.grid()
    t1= np.linspace(-0.01, 0.4, 1000)
    plt.plot(t1, step(t1)*5.4, color="b", linestyle="--", label = "ステップ信号")
    plt.plot(T, yout, color = "red", label = "シミュレーション結果")
    plt.xlim(-0.01, 0.4)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    plt.show()

def step(x):
  return 1.0 * (x >= 0.0)

if __name__ == "__main__":
    v_in_a_spe()
    v_out_a_spe()
    v_in_b_spe()
    v_out_b_spe()
    v_in_b_pos()
    v_out_b_pos()


