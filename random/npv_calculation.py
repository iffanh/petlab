import numpy as np
import pandas as pd

import json
import sys

import common

class NPVDataSet:

    def __init__(self, npv_file):

        with open(npv_file, 'r') as f:
            npv_data = json.load(f)

        self.total_year = npv_data['Years']
        self.uptime = npv_data['Uptime']
        self.capex = self.Capex(npv_data['CAPEX'], self.total_year)
        self.opex = self.Opex(npv_data['OPEX'])
        self.drillex = self.Drillex(npv_data['DRILLEX'], self.total_year)
        self.discount_rate = npv_data['Discount_rate']
        self.oil_price = npv_data['Oil_price']
        self.fopt = self.FOPT(npv_data['FOPT'])

    def get_capex(self):

        return self.Capex(self.capex)

    class Capex:
        def __init__(self, capex, total_year):
            self.start_year = capex['Start_year']
            self.fraction = capex['Fraction']

            # Check fraction
            temp = 0
            for frac in self.fraction:
                temp += frac
            if temp > 1:
                raise ValueError('Capex fraction aggregation is more than 1!')

            # Check start_year
            if self.start_year > total_year :
                raise ValueError('Capex start year is beyond the total year')
            elif self.start_year <= 0 :
                raise ValueError('Capex start year has to be positive')
            else:
                pass

            # Check length
            if self.start_year + len(self.fraction) - 1 <= total_year:
                pass
            else:
                raise ValueError('Capex fraction is too long or total year is too short')


            self.amount = capex['Amount']

    class Opex:
        def __init__(self, opex):
            self.start_year = opex['Start_year']
            self.amount = opex['Amount']

    class Drillex:
        def __init__(self, drillex, total_year):
            self.start_year = drillex['Start_year']
            self.excalation = drillex['Excalation']
            self.amount = drillex['Amount']
            self.wells = drillex['Wells']

            # Check start_year
            if self.start_year > total_year :
                raise ValueError('Drillex start year is beyond the total year')
            elif self.start_year <= 0 :
                raise ValueError('Drillex start year has to be positive')
            else:
                pass

            # Check length
            if self.start_year + len(self.wells) - 1 <= total_year:
                pass
            else:
                raise ValueError('Drillex well planning is too long or total year is too short')


    class FOPT:
        def __init__(self, fopt):
            self.start_year = fopt['Start_year']
            self.data = fopt['Data']

def get_production_total(total_year, start_year):

    # Set production data to match the starting year
    fopt_data = np.load(npv.fopt.data)

    # Input fopt data to the data frame
    fopt_list = []
    for ii in range(1, total_year + 1):
        if ii < start_year:
            fopt_list.append(0.0)
        else:
            fopt_list.append(fopt_data[ii-start_year])

    return fopt_list

def get_revenue(fopt, oil_price):

    return fopt*npv.oil_price

def get_drillex(total_year, start_year, amount, excalation, wells):

    drillex_data = np.zeros(total_year)

    for i in range(start_year, start_year + len(wells)):
        drillex_data[i-1] = wells[i-start_year]*amount*((1+excalation)**i)

    return drillex_data

def get_capex(total_year, start_year, amount, fraction):

    capex_data = np.zeros(total_year)

    for i in range(start_year, start_year + len(fraction)):
        capex_data[i-1] = fraction[i-start_year]*amount

    return capex_data

def get_opex(total_year, start_year, amount):

    opex_data = np.zeros(total_year)

    for i in range(start_year, total_year + 1):
        opex_data[i-1] = amount

    return opex_data

def get_discounted_cash_flow(total_year, cash_flow, discounted_rate):

    dcf_data = np.zeros(total_year)
    cash_flow = np.array(cash_flow)
    for i in range(1, total_year+1):
        dcf_data[i-1] = cash_flow[i-1]/((1 + discounted_rate)**i)

    return dcf_data

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('To run: python npv_calculation.py /path/to/npv_dataset.json/ /path/to/result.csv/')
        sys.exit(1)

    file_path = sys.argv[1]
    result_path = sys.argv[2]

    npv = NPVDataSet(file_path)

    uptime_days = npv.uptime * 365

    # Initialize dataframe
    df = pd.DataFrame(index=range(1,npv.total_year+1))

    df['FOPT (stb)'] = get_production_total(npv.total_year, npv.fopt.start_year)
    df['Revenue (USD)'] = get_revenue(df['FOPT (stb)'], npv.oil_price)
    df['Drillex (USD)'] = get_drillex(npv.total_year, npv.drillex.start_year, npv.drillex.amount, npv.drillex.excalation, npv.drillex.wells)
    df['Capex (USD)'] = get_capex(npv.total_year, npv.capex.start_year, npv.capex.amount, npv.capex.fraction)
    df['Opex (USD)'] = get_opex(npv.total_year, npv.opex.start_year, npv.opex.amount)
    df['Total_Cost (USD)'] = df['Drillex (USD)'] + df['Capex (USD)'] + df['Opex (USD)']
    df['Cash_Flow (USD)'] = df['Revenue (USD)'] - df['Total_Cost (USD)']
    df['DCF (USD)'] = get_discounted_cash_flow(npv.total_year, df['Cash_Flow (USD)'], npv.discount_rate)

    total_wells = np.sum(npv.drillex.wells)
    net_present_value = df['DCF (USD)'].sum()


    print(df)
    # Writing results to the path

    common.write_to_csv('NPV Calculation', result_path)
    common.append_to_csv('Dataset source, %s \n' %file_path, result_path )
    common.append_to_csv('Total number of years, %s, Years \n' %npv.total_year, result_path )
    common.append_to_csv('Uptime, %s \n' %npv.uptime, result_path )
    common.append_to_csv('Discount rate, %s \n' %npv.discount_rate, result_path )
    common.append_to_csv('Oil price, %s, USD/stb \n' %npv.oil_price, result_path )
    common.append_to_csv('Total number of wells, %s \n \n' %total_wells, result_path )

    df.to_csv(result_path, mode='a')
    # data = [34700000., 33700000., 26700000., 20100000., 15300000., 11600000.,  8720000.,
    #         6620000.,  5010000.,  3730000.,  2760000.,  2100000.,  1590000.,   1150000.]

    common.append_to_csv('\n', result_path )
    common.append_to_csv('Net Present Value (NPV), %s, USD \n' %net_present_value, result_path )
    # np.save("./json_files/npv_calculation/FOPT.npy", data)