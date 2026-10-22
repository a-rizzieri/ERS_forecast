import numpy as np
import pysm3.units as u



def jysr2cmb(freqs):

    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def symmetrized_pairs(n):
    return [(i, j) for i in range(n) for j in range(i, n)]
