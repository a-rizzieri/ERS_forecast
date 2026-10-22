"""
Classes and functions to reproduce the number counts of ERS at 5 GHz frequency
Total and flat+inv
"""
import numpy as np
from dataclasses import dataclass

# Differential number counts at 5 GHz for the 3 populations:
# - ncounts for flat+inverted: from Tucci+11 Eq. (1) fitted on data 
# - ncounts for total (steep+flat+inverted): Toffolatti et al. 1998


@dataclass(frozen=True)
class TucciFlatInvDiffNumCounts5GHz:
    """
    Tucci et al. 2011 Eq. (1) fit for the combined flat+inverted
    differential number counts at 5 GHz.
    """
    n0: float = 47.4   # Jy^-1 sr^-1
    S0: float = 1.67   # Jy
    k: float = 0.50
    l: float = -0.66

    def nbar(self, S_jy: np.ndarray | float) -> np.ndarray:
        """
        Euclidean-normalized differential counts:
            nbar(S) = S^(5/2) dN/dS
        """
        S = np.asarray(S_jy, dtype=float)
        x = S / self.S0
        return self.n0 * (x ** self.k) / (1. - np.exp(-1.)) * (1. - np.exp(-(x ** (self.l - self.k))))

    def dNdS(self, S_jy: np.ndarray | float) -> np.ndarray:
        """
        Differential counts dN/dS in Jy^-1 sr^-1.
        (note this is indeed dN/dS, not dN/dlog10S)
        """
        S = np.asarray(S_jy, dtype=float)
        return self.nbar(S) * S ** (-2.5)


@dataclass(frozen=True)
class TotalDiffNumCounts5GHz:
    """
    Total 5 GHz differential number counts from Toffolatti Fig. 1 (solid curve).
    The digitized data are stored in the data/ directory.

    CSV format:
        column 0: log10(S)   [Jy]
        column 1: log10((dN/dS) / Delta_N0)

    with
        Delta_N0 = 150 * S^(-2.5)   [Jy^-1 sr^-1]

    Therefore:
        dN/dS = Delta_N0 * 10**(log10((dN/dS) / Delta_N0))   [Jy^-1 sr^-1]
    """
    log10_S: np.ndarray
    log10_ratio: np.ndarray  # ratio stands for : (dN/dS) / Delta_N0

    @classmethod
    def from_csv(
        cls,
        path: str,
        delimiter: str = ",",
        skiprows: int = 1,
    ) -> "TotalDiffNumCounts5GHz":
        data = np.loadtxt(
            path,
            delimiter=delimiter,
            comments="#",
            skiprows=skiprows,
        )

        if data.shape[1] != 2:
            raise ValueError(
                "CSV must have two columns: log10(S), log10((dN/dS) / Delta_N0)."
            )

        log10_S = data[:, 0]
        log10_ratio = data[:, 1]

        order = np.argsort(log10_S)

        return cls(
            log10_S=log10_S[order],
            log10_ratio=log10_ratio[order],
        )

    def _interp_log10_ratio(self, S_jy: np.ndarray | float) -> np.ndarray:
        """
        Interpolates the number counts loaded from disk 
        at the requested flux densities S_jy.
        """
        S_input = np.asarray(S_jy, dtype=float)

        S = np.atleast_1d(S_input)

        if np.any(S <= 0):
            raise ValueError("Flux densities must be strictly positive.")

        # When I have S values outside the original given in Toffolatti+98 Fig. 1, 
        # I extrapolate with a linear function in log-log space.
        x = np.log10(S)
        xp = self.log10_S
        yp = self.log10_ratio

        y = np.interp(x, xp, yp)

        # left extrapolation
        left = x < xp[0]
        if np.any(left):
            m_left = (yp[1] - yp[0]) / (xp[1] - xp[0])
            y[left] = yp[0] + m_left * (x[left] - xp[0])

        # right extrapolation
        right = x > xp[-1]
        if np.any(right):
            m_right = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
            y[right] = yp[-1] + m_right * (x[right] - xp[-1])

        return y


    def dNdS(self, S_jy: np.ndarray | float) -> np.ndarray:
        """
        Differential counts dN/dS in Jy^-1 sr^-1.
        (note this is indeed dN/dS, not dN/dlog10S)
        """
        S = np.asarray(S_jy, dtype=float)
        log10_ratio = self._interp_log10_ratio(S)

        delta_N0 = 150.0 * S ** (-2.5)

        out = delta_N0 * (10.0 ** log10_ratio)

        return out
