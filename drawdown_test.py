import numpy as np
import pandas as pd
import json
import scipy.stats

import matplotlib.pyplot as plt

import sys
from functools import partial

import common

class DrawdownAnalysis:

    def __init__(self, dd_file):

        with open(dd_file, 'r') as f:
            dd_data = json.load(f)

        self.rate = dd_data['Rate']
        self.viscosity = dd_data['Viscosity']
        self.porosity = dd_data['Porosity']
        self.pressure = dd_data['Pressure']
        self.well_radius = dd_data['Well_Radius']
        self.reservoir_height = dd_data['Reservoir_Height']
        self.total_compressibility = dd_data['Total_Compressibility']
        self.pressure_data = dd_data['Pressure_Data']
        self.time_data = dd_data['Time_Data']

        if len(self.pressure_data) == len(self.time_data):
            pass
        else:
            raise ValueError("Length of pressure (%d) and time data (%d) not match. " %(len(self.pressure_data), len(self.time_data)))


def sort_list_if_not_empty(index_list):
    if len(index_list) == 0:
        pass
    else:
        index_list.sort()

def calculate_permeability(data, slope):

    return (-1* data.rate * data.viscosity / (4 * np.pi * data.reservoir_height * slope ))

def onpick(event, data, indx, non_indx, dots, line_fit, ax, result_path):

    N = len(event.ind)
    if not N: return True

    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind


    if ind in indx:
        indx.remove(ind)
        non_indx.extend(ind)

        sort_list_if_not_empty(indx)
        sort_list_if_not_empty(non_indx)

    else:
        non_indx.remove(ind)
        indx.extend(ind)

        sort_list_if_not_empty(indx)
        sort_list_if_not_empty(non_indx)

    ax.cla()

    draw_drawdown_plot(data, indx, non_indx, ax, result_path)

    plt.legend()
    plt.draw()

    return True

def draw_drawdown_plot(data, indx, non_indx, ax, result_path):

    data_x_on = np.array(data.time_data)[indx]
    data_y_on = np.array(data.pressure_data)[indx]

    data_x_off = np.array(data.time_data)[non_indx]
    data_y_off = np.array(data.pressure_data)[non_indx]

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(data_x_on), data_y_on)

    y_init = slope*np.log(data.time_data[0]) + intercept
    y_last = slope*np.log(data.time_data[-1]) + intercept

    x_fit = [data.time_data[0], data.time_data[-1]]
    y_fit = [y_init, y_last]

    # y_init = slope*np.log(data_x_on[0]) + intercept
    # y_last = slope*np.log(data_x_on[-1]) + intercept

    # x_fit = [data_x_on[0], data_x_on[-1]]
    # y_fit = [y_init, y_last]

    color_list = ['g' if i in indx else 'b' for i, d in enumerate(data.time_data)]

    line_fit = ax.plot(x_fit, y_fit, 'g--')
    dots_on, = ax.plot(data_x_on, data_y_on, 'o', c = 'g', label='On')
    dots_off, = ax.plot(data_x_off, data_y_off, 'o', c = 'r', label='Off')
    dots, = ax.plot(data.time_data, data.pressure_data, 'o', picker=5, c = 'w', alpha=0.01)
    ax.set_title("Pressure vs time. Toggle the dots on/off by clicking on them.")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pressure (Pa)")
    ax.set_xscale('log')

    print('intercept = %s' %intercept)
    print('slope = %s' %slope)
    print('r-value = %s' %r_value)
    print('p-value = %s' %p_value)

    # Calculate permeability
    perm = calculate_permeability(data, slope)

    perm_in_md = perm * 1.01325E+15

    print('Permeability = %s m2' %perm)
    print('Permeability = %s mD' %perm_in_md)

    common.write_to_csv('Intercept, %s \n' %intercept, result_path)
    common.append_to_csv('slope, %s, Pa \n' %slope, result_path)
    common.append_to_csv('r-value, %s \n' %r_value, result_path)
    common.append_to_csv('p-value, %s \n' %p_value, result_path)
    common.append_to_csv('Permeability, %s, m2 \n' %perm, result_path)
    common.append_to_csv('Permeability, %s, mD \n' %perm_in_md, result_path)


    return dots, line_fit

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('To run: python drawdown_test.py /path/to/drawdown_dataset.json/ /path/to/result.csv/')
        sys.exit(1)

    file_path = sys.argv[1]
    result_path = sys.argv[2]

    # fileName = './json_files/dd_dataset_1.json'

    data = DrawdownAnalysis(file_path)

    ii_init = 0
    ii_last = len(data.time_data)

    indx = [x for x in range(ii_init, ii_last)]
    non_indx = []

    fig, ax = plt.subplots(1, 1, figsize=(16,9))

    dots, line_fit = draw_drawdown_plot(data, indx, non_indx, ax, result_path)

    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, data, indx, non_indx, dots, line_fit, ax, result_path))
    plt.legend()
    plt.show()