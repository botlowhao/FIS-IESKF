#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from vwio_eskf.msg import WOFISData, Q1Data

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from custom_mf import custom_mf2, custom_mf3


def data_preprocessing(file_path):
    data = pd.read_csv(file_path)
    rho = data.iloc[:, 0]
    omega = data.iloc[:, 1]

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

    return rho_stats, omega_stats

def create_fis(rho_stats, omega_stats):
    Am1 = rho_stats["max"]
    Am2 = rho_stats["max"]
    Am3 = rho_stats["min"]
    Am4 = rho_stats["min"]
    Am5 = omega_stats["min"]
    Am6 = omega_stats["max"]
    Am7 = omega_stats["min"]
    Am8 = omega_stats["max"]

    x_rho = np.arange(-25.0000, 26.0001, 0.0001)
    x_omega = np.arange(-0.6000, 0.6001, 0.0001)
    x_Q1 = np.arange(0, 1.0001, 0.0001)

    rho = ctrl.Antecedent(x_rho, 'p')
    omega = ctrl.Antecedent(x_omega, 'w')
    Q1 = ctrl.Consequent(x_Q1, 'Q1')

    rho['A1'] = fuzz.smf(x_rho, -1, Am1)
    rho['A2'] = custom_mf2(x_rho, 1, Am2)
    rho['A3'] = custom_mf3(x_rho, Am3, -1)
    rho['A4'] = fuzz.zmf(x_rho, Am4, 1)
    omega['A5'] = fuzz.zmf(x_omega, Am5, Am6)
    omega['A6'] = fuzz.smf(x_omega, Am7, Am8)

    Q1['High Variance'] = fuzz.trimf(x_Q1, [0.33, 0.66, 1])
    Q1['Low Variance'] = fuzz.trimf(x_Q1, [0, 0.33, 0.66])

    Q1.defuzzify_method = 'centroid'

    rule1 = ctrl.Rule(antecedent=(rho['A1'] & omega['A6']), consequent=Q1['High Variance'], label='R1')
    rule2 = ctrl.Rule(antecedent=(rho['A2'] & omega['A6']), consequent=Q1['High Variance'], label='R2')
    rule3 = ctrl.Rule(antecedent=(rho['A4'] & omega['A6']), consequent=Q1['High Variance'], label='R3')
    rule4 = ctrl.Rule(antecedent=(rho['A1'] & omega['A5']), consequent=Q1['High Variance'], label='R4')
    rule5 = ctrl.Rule(antecedent=(rho['A4'] & omega['A5']), consequent=Q1['High Variance'], label='R5')
    rule6 = ctrl.Rule(antecedent=(rho['A3'] & omega['A6']), consequent=Q1['Low Variance'], label='R6')
    rule7 = ctrl.Rule(antecedent=(rho['A2'] & omega['A5']), consequent=Q1['Low Variance'], label='R7')
    rule8 = ctrl.Rule(antecedent=(rho['A3'] & omega['A5']), consequent=Q1['High Variance'], label='R8')

    Q1_Calculate_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
    Q1_Calculate = ctrl.ControlSystemSimulation(Q1_Calculate_ctrl)

    return Q1_Calculate

class FISWOFISNode:
    def __init__(self):
        rospy.init_node('fis_wo_node')
        
        self.subscriber = rospy.Subscriber('WOFIS_msg', WOFISData, self.wofis_data_callback)
        self.publisher = rospy.Publisher('Q1_msg', Q1Data, queue_size=100)
        
        file_path = '/home/wyatt/catkin_ws/src/fis_wo/FIS_dateset/ANFIS_WO_sync8.csv'
        rho_stats, omega_stats = data_preprocessing(file_path)
        self.Q1_Calculate = create_fis(rho_stats, omega_stats)
        
    def wofis_data_callback(self, msg):
        self.Q1_Calculate.input['p'] = msg.delta_v
        self.Q1_Calculate.input['w'] = msg.w_z
        self.Q1_Calculate.compute()
        
        output_Q1 = self.Q1_Calculate.output['Q1']
        # rospy.loginfo(f"Publishing Q1: {output_Q1}")
        
        Q1_msg = Q1Data()
        Q1_msg.q1 = output_Q1
        self.publisher.publish(Q1_msg)
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = FISWOFISNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
