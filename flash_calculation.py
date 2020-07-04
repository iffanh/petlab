import numpy as np
import pandas as pd
import json
import scipy.optimize

import sys
from functools import partial

class PVT:


    def __init__(self, componentFile):
        ## Constant:
        self.gas_constant = 8.31446261815324

        with open(componentFile, 'r') as f:
            fluidData = json.load(f)

        self.component = fluidData['component']

        self.pressureK = fluidData['PressureK']
        self.pressure = fluidData['Pressure']
        self.temperature = fluidData['Temperature']
        self.fv = fluidData['fv']
        self.max_iter = fluidData['max_iter']
        self.A0 = fluidData['A0']

        ## Calculation
        if hasattr(self, 'A1'):
            pass
        else:
            try:
                self.A1 = 1 - ((self.pressure - 14.7)/(self.pressureK - 14.7))**self.A0
            except:
                raise ValueError("Not enough information to calculation A1. Give A0 information in the fluid .json file")



    def get_component(self):
        return [self.Component(comp, self.component[comp]) for comp in self.component.keys()]

    def update_fv(self, fv):
        self.fv = fv

    class Component:

        def __init__(self, comp, component):
            self.name = comp
            self.mole_fraction = component['Mole_Fraction']
            self.critical_pressure = component['Critical_Pressure']
            self.critical_temperature = component['Critical_Temperature']
            self.accentric_factor = component['Accentric_Factor']


    def get_A1(self):
        return 1 - ((self.pressure - 14.7)/(self.pressureK - 14.7))**self.A0


def calculate_k_value(pk, pc, tc, a1, w, p, t):
    K = ((pc/pk)**(a1-1))*(np.exp(5.37*a1*(1+w)*(1-(t/tc)**(-1))))/(p/pc)
    return K

def calculate_c(k):
    if k == 1.0:
        return 0
    else:
        return 1/(k-1)

def calculate_h(fv, c, z):
    if c == 0.0:
        return 0
    else:
        return (z/(fv + c))

def objective_function_for_fv(fv, zList, cList):

    total = 0
    for i, z in enumerate(zList):
        pfunction = partial(calculate_h, c = cList[i], z = zList[i])
        total += pfunction(fv)

    return abs(total)

def optimize_fv(fluid, df):
    return scipy.optimize.minimize(objective_function_for_fv, fluid.fv, args=(df['Mole_Fraction'].to_numpy(), df['c_i'].to_numpy()))

def fill_table(fluid, df):

    kvalues = []
    cvalues = []
    hvalues = []
    for ii in range(len(df.index)):
        ## Calculate K-Value
        k = calculate_k_value(fluid.pressureK,
                          fluid.get_component()[ii].critical_pressure,
                          fluid.get_component()[ii].critical_temperature,
                          fluid.A1,
                          fluid.get_component()[ii].accentric_factor,
                          fluid.pressure,
                          fluid.temperature
                          )
        kvalues.append(k)

        ## Calculate c
        c = calculate_c(k)
        cvalues.append(c)

        ## Calculate h
        h = calculate_h(fluid.fv, c, fluid.get_component()[ii].mole_fraction)
        hvalues.append(h)

    df['K_Values'] = kvalues
    df['c_i'] = cvalues
    df['h_i'] = hvalues

    return df

def calculate_x(z, fv, k):
    return (z/(fv*(k - 1) + 1))

def calculate_y(x, k):
    return x*k

def fill_liquid_and_gas_comp_table(df, fluid):
    # Calculate x and y
    xvalues = []
    yvalues = []

    for ii in range(len(df.index)):
        x = calculate_x(fluid.get_component()[ii].mole_fraction, fluid.fv, df['K_Values'].to_numpy()[ii])
        xvalues.append(x)

        y = calculate_y(x, df['K_Values'].to_numpy()[ii])
        yvalues.append(y)

    df['x_i'] = xvalues
    df['y_i'] = yvalues

    return df

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('To run: python flash_calculation.py /path/to/fluid.json/ /path/to/result.csv/')
        sys.exit(1)

    fluid_path = sys.argv[1]

    # fluid = PVT("./pvt_dataset_1.json")
    fluid = PVT(fluid_path)

    # Check compositions to add up to 1
    totalZ = 0
    for comp in fluid.get_component():
        totalZ += comp.mole_fraction

    if abs(totalZ - 1) < 10**-4:
        print("Mole fraction adds up to 1")
        pass
    else:
        raise ValueError("Mole fraction adds up to %s and does not add up to 1" %totalZ)


    # Write the dataframe
    # resultPath = "./results/pvt_results.csv"
    resultPath = sys.argv[2]
    df = pd.DataFrame()
    df['Component'] = [x.name for x in fluid.get_component()]
    df['Mole_Fraction'] = [x.mole_fraction for x in fluid.get_component()]
    df['Critical_Pressure'] = [x.critical_pressure for x in fluid.get_component()]
    df['Critical_Temperature'] = [x.critical_temperature for x in fluid.get_component()]
    df['Accentric_Factor'] = [x.accentric_factor for x in fluid.get_component()]


    # Initializing table
    df = fill_table(fluid, df)

    # Check if h sum is zero or not
    iter = 0
    while abs(np.sum(df['h_i'].to_numpy())) > 10**-8:

        if iter > fluid.max_iter:
            print(df)
            raise RuntimeError("Exceeded maximum iteration of %d to calculate fv" %fluid.max_iter)
            # break
        # Get new fv
        fluid.update_fv(optimize_fv(fluid, df).x[0])
        # Refill table
        df = fill_table(fluid, df)

        iter += 1


    df = fill_liquid_and_gas_comp_table(df, fluid)

    df.to_csv(resultPath, index = False, header=True)

    print('SUCCESS: Flash calculation success')




