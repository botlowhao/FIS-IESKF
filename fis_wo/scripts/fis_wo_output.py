#!/usr/bin/env python3

import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from custom_mf import Inverted_gbellmf

from matplotlib import rcParams

config = {
    "font.family": 'serif',          # 使用 serif 字体
    "font.size": 28,                 # 字体大小14，相当于小四
    "mathtext.fontset": 'stix',      # 数学字体设置为 stix，接近 Times New Roman
    'axes.unicode_minus': False,     # 处理负号的显示
    'axes.labelweight': 'bold',      # 坐标轴标签加粗
    'lines.linewidth': 3,            # 线条加粗为 2
    'axes.titlesize': 32,            # 标题字体大小
    'axes.titleweight': 'bold',      # 标题加粗
    'axes.labelsize': 27,            # 坐标轴标签字体大小
    'legend.fontsize': 28            # 图例字体大小
}
rcParams.update(config)



def data_preprocessing(file_path):
    # Read the FIS data
    data = pd.read_csv(file_path)

    # Extract columns
    rho = data.iloc[:, 0]
    omega = data.iloc[:, 1]
    v_x = data.iloc[:, 2]


    # Calculate statistical values
    rho_stats = {
        "min": np.min(rho),
        "max": np.max(rho),
        "mean": np.mean(rho),
        "std": np.std(rho)
    }

    omega_stats = {
        "min": np.min(omega),
        "max": np.max(omega),
        "mean": np.mean(omega),
        "std": np.std(omega)
    }

    vx_stats = {
        "min": np.min(v_x),
        "max": np.max(v_x),
        "mean": np.mean(v_x),
        "std": np.std(v_x)
    }

    return rho_stats, omega_stats, vx_stats

def create_fis(rho_stats):
    # Set data for Am1-Am8
    Am1 = rho_stats["max"]
    Am2 = rho_stats["max"]
    # Am3 = rho_stats["min"]
    Am3 = rho_stats["min"]
    # Am4 = omega_stats["min"]
    # Am5 = omega_stats["max"]
    # Am6 = omega_stats["min"]
    # Am7 = omega_stats["max"]

    # Add Input fuzzy variable p, w
    x_rho = np.arange(-25.0000, 26.0001, 0.0001)
    x_omega = np.arange(-0.6000, 0.6001, 0.0001)
    x_vx = np.arange(-0.1000, 0.1000, 0.0001)
    x_Q1 = np.arange(0, 1.0001, 0.0001)

    # Define fuzzy control variable
    rho = ctrl.Antecedent(x_rho, 'p')
    omega = ctrl.Antecedent(x_omega, 'w')
    v_x = ctrl.Antecedent(x_vx, 'v_x')
    Q1 = ctrl.Consequent(x_Q1, 'Q1')

    # Define input fuzzy membership function
    rho['A1'] = fuzz.smf(x_rho, 1, Am1)
    rho['A2'] = fuzz.gaussmf(x_rho, 0.0, 0.1)
    # rho['A3'] = custom_mf3(x_rho, Am3, -1)
    rho['A3'] = fuzz.zmf(x_rho, Am3, -1)
    # omega['A4'] = fuzz.zmf(x_omega, Am4, Am5)
    omega['A4'] = fuzz.gbellmf(x_omega, 0.3, 4, 0)
    # omega['A5'] = fuzz.smf(x_omega, Am6, Am7)
    omega['A5'] = Inverted_gbellmf(x_omega, 0.3, 4, 0)
    v_x['A6'] = fuzz.gaussmf(x_vx, 0.00, 0.0035)

    # Define output fuzzy membership function
    Q1['High Variance'] = fuzz.trimf(x_Q1, [0.33, 0.66, 1])
    Q1['Low Variance'] = fuzz.trimf(x_Q1, [0, 0.33, 0.66])

    # Set defuzzification method 
    Q1.defuzzify_method = 'centroid'

    # Display input/output variable and its membership function
    
    # rho.view() 只能用红绿蓝
    plt.plot(x_rho, rho['A1'].mf, 'r', label='A1')
    plt.plot(x_rho, rho['A2'].mf, 'g', label='A2')
    plt.plot(x_rho, rho['A3'].mf, 'b', label='A3')
    plt.title(r"WO-FIS Input Membership Function $\rho$")
    plt.xlabel(r"$\rho$")
    plt.ylabel("Membership Value")
    plt.legend()
    plt.show()


    # omega.view() 只能用红蓝
    plt.plot(x_omega, omega['A4'].mf, 'r', label='A4')
    plt.plot(x_omega, omega['A5'].mf, 'b', label='A5')
    plt.title(r"WO-FIS Input Membership Function $\omega_e^z$")
    plt.xlabel(r"$\omega_e^z$")
    plt.ylabel("Membership Value")
    plt.legend()
    plt.show()

    # v_x.view() 只能用蓝色
    plt.plot(x_vx, v_x['A6'].mf, 'b', label='A6')
    plt.title(r"WO-FIS Input Membership Function $v_e^x$")
    plt.xlabel(r"$v_e^x$")
    plt.ylabel("Membership Value")
    plt.legend()
    plt.show()

    # Q1.view() 只能用红蓝
    plt.plot(x_Q1, Q1['High Variance'].mf, 'r', label='$W_{\mathrm{H}}$')
    plt.plot(x_Q1, Q1['Low Variance'].mf, 'b', label='$W_{\mathrm{L}}$')
    plt.title(r"WO-FIS Output Membership Function $\hat{\sigma}_{\mathbf{p}}^{2}$")
    plt.xlabel(r"$\hat{\sigma}_{\mathbf{p}}^{2}$")
    plt.ylabel("Membership Value")
    plt.legend()
    plt.show()

    # Set fuzzy rules
    rule1 = ctrl.Rule(antecedent=(rho['A1'] & omega['A5']), consequent=Q1['High Variance'], label='R1')
    rule2 = ctrl.Rule(antecedent=(rho['A2'] & omega['A5']), consequent=Q1['High Variance'], label='R2')
    rule3 = ctrl.Rule(antecedent=(rho['A3'] & omega['A5']), consequent=Q1['High Variance'], label='R3')
    rule4 = ctrl.Rule(antecedent=(rho['A1'] & omega['A4']), consequent=Q1['High Variance'], label='R4')
    rule5 = ctrl.Rule(antecedent=(rho['A3'] & omega['A4']), consequent=Q1['High Variance'], label='R5')
    # rule6 = ctrl.Rule(antecedent=(rho['A3'] & omega['A6']), consequent=Q1['Low Variance'], label='R6')
    rule6 = ctrl.Rule(antecedent=(rho['A2'] & omega['A4']), consequent=Q1['Low Variance'], label='R6')
    # rule8 = ctrl.Rule(antecedent=(rho['A3'] & omega['A5']), consequent=Q1['High Variance'], label='R8')
    rule7 = ctrl.Rule(antecedent=(omega['A4'] & v_x['A6']), consequent=Q1['Low Variance'], label='R7')
    rule8 = ctrl.Rule(antecedent=(omega['A5'] & v_x['A6']), consequent=Q1['High Variance'], label='R8')


    Q1_Calculate_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
    Q1_Calculate = ctrl.ControlSystemSimulation(Q1_Calculate_ctrl)

    return Q1_Calculate, Q1

def test_fis(Q1_Calculate):
    # Test for output Q1
    Q1_Calculate.input['p'] = 5.0000
    Q1_Calculate.input['w'] = 0.2000
    Q1_Calculate.input['v_x'] = 0.03
    Q1_Calculate.compute()
    return Q1_Calculate.output['Q1']

if __name__ == "__main__":
    file_path = '/home/wyatt/catkin_ws/src/fis_wo/FIS_dateset/FIS_WO_81.csv'
    rho_stats, omega_stats,vx_stats = data_preprocessing(file_path)
    # print(vx_stats)
    # print(omega_stats)
    # print(rho_stats)
    Q1_Calculate, Q1 = create_fis(rho_stats)
    output_Q1 = test_fis(Q1_Calculate)
    print(output_Q1)
    Q1.view(sim=Q1_Calculate)
    plt.show()
