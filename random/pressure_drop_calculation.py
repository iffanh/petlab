import numpy as np
import pandas as pd

import json
import sys

import common


class BeggsBrill:

    def __init__(self, p_in, Q_l, Q_g, v, D, h, theta, nz):
        # From https://cheguide.com/beggs_brill.html
        # Input parameters
        self.p_in = p_in
        self.Q_l = Q_l
        self.Q_g = Q_g
        self.v = v
        self.D = D
        self.h = h
        self.theta = theta
        self.nz = nz

        # Preprocess input data

        self.dz = h/nz

        # Constants
        self.gc = 9.8
        self.g = 9.8 

    def _calculate_froude_number(self):
        return self.v**2 / (self.g * self.D)

    def _calculate_liquid_fraction(self):
        return self.Q_l / (self.Q_l + self.Q_g)

    def run(self):
        
        print('Begin pressure drop calculation')

        for section in range(nz):

            # Calculate Froude Number
            Fr = self._calculate_froude_number()

            # Calculate C_l
            C_l = self._calculate_liquid_fraction()

            # Calculate L1, L2, L3, and L4

class PressureDropCalculation:

    def __init__(self):
        pass
        
    def beggs_brill(self, p_in, Q_l, Q_g, v, D, h, theta, nz=10):
        
        """
        Parameters:
        p_in    : inflow pressure 
        Q_l     : liquid volumetric flow
        Q_g     : gas volumetric flow
        v       : mixture flow rate
        D       : pipe diameter
        h       : pipe length
        theta   : pipe angle
        nz      : number of sections
        """
        self.pressure_drop = BeggsBrill(p1, p2, v, D, h, theta, nz)
