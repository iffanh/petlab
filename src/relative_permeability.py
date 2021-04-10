import numpy as np 
import os 

class RelativePermeability():

    def __init__(self):

        return

class CoreyCorrelation():

    def __init__(self, sl, sr, krw, krnw, nw, nnw):

        """
        Args:
            sl (float)  : largest saturation in which the phase starts moving
            sr (float)  : smallest saturation in which the phase reach its maximum 
                    relative permeability
            krw (float) : wetting phase end-point relative permeability
            krnw (float): non-wetting phase end-point relative permeability
            nw (float)  : wetting phase Corey exponent
            nnw (float) : non-wetting phase Corey exponent

        """

        # Check input
        ## saturation must be between 0 and 1
        for sat in [sl, sr]:
            if sat > 1.0 or sat < 0.0:
                raise Exception("Saturation values must be between 0 and 1. Found %s" %sat)
        
        ## sl + sr must be < 1.0
        if sl + sr > 1.0:
            raise Exception("sl + sr must be less than 1.0")

        ## relative permeability endpoint must be positive
        for kr in [krw, krnw]:
            if kr < 0.0:
                raise Exception("Relative permeability end point must be a positive value. Found %s" %kr)

        ## corey exponent must be between 1 and 6
        for n in [nw, nnw]:
            if n < 1.0 or n > 6.0:
                raise Exception("Corey exponent must be between 1.0 and 6.0. Found %s" %n)

        # Save input 
        self.sl = sl
        self.sr = sr 
        self.krw = krw 
        self.krnw = krnw 
        self.nw = nw 
        self.nnw = nnw 

        # sf for saturation function
        self.krw = lambda s: self._normalized_rperm(s, sl, sr, krw, nw)
        self.krnw = lambda s: self._normalized_rperm((1-s), sr, sl, krnw, nnw)

        return

    def _normalized_saturation(self, s, sl, sr):
        """ This function normalizes the saturation based on its
        'irreducible' values

        Args: 
            s (float)   : saturation
            sl (float)  : largest saturation in which the phase starts moving
            sr (float)  : smallest saturation in which the phase reach its maximum 
                    relative permeability

        Returns:
            sn (float)  : normalized saturation
        """
        if s < sl:
            sn = 0
        elif s > 1-sr:
            sn = 1
        else:
            sn = (s - sl)/(1-sl-sr)

        return sn

    def _normalized_rperm(self, s, sl, sr, k_end, exponent):

        """This function returns the value of relative permeability as a function
        of saturation (s), given its 'irreducible' saturations.

        Original
        0|---sl---------s-----(1-sr)---|1

        Normalized (shift sl to 0 and 1-sr to 1)
        0|-----------------sn----------|1

        w refers to wetting phase (not to be confused with water)
        nw refers to non-wetting phase

        Args: 
            s (float)   : saturation
            sl (float)  : largest saturation in which the phase starts moving
            sr (float)  : smallest saturation in which the phase reach its maximum 
                    relative permeability
            k_end (float)  : relative permeability endpoint
            exponent (float)    : saturation exponent

        Returns:
            kr (float)  : relative permeability values

        """
        sn = self._normalized_saturation(s, sl, sr)
        kr = k_end*(sn)**exponent
        return kr