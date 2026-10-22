"""
Fully analytical model for extragalactic radio sources based on Tucci et al. (2011).

The module computes integral number counts and polarised Poisson auto- and
cross-spectra at arbitrary frequencies, starting from the 5 GHz number counts.
High-frequency fluxes are obtained by integrating over the population-dependent
spectral-parameter distributions using deterministic numerical quadrature.

The model includes steep-, flat-, and inverted-spectrum populations, with the C0
prescription for flat-spectrum sources, and optional source masking
with residual Gaussian wings.

Main functions:
- tucci_cross_spectra()
- tucci_number_counts()
both based on the _TucciSpectralQuadrature class, which implements the shared 
deterministic numerical quadrature over the spectral parameters.
"""
import numpy as np
from scipy.stats import norm, truncnorm
from utils import jysr2cmb


_ALPHA_BINS = [
    # Gaussian spectral index distributions at 5 GHz, Tucci et al 2011 Table 1
    # Smin [Jy], Smax [Jy], steep_mean, steep_sigma, flat+inv_mean, flat+inv_sigma
    (0.100, 0.158, -0.87, 0.15, -0.39, 0.38),
    (0.158, 0.251, -0.86, 0.14, -0.29, 0.37),
    (0.251, 0.400, -0.86, 0.14, -0.17, 0.30),
    (0.400, np.inf, -0.81, 0.14, -0.14, 0.40),
]

_ALLOWED_POPULATIONS = {"total", "steep", "flat", "inverted", "flatinv", "all"}
_ALLOWED_FLAT_MODELS = {"C0"}


def _alpha_bin_params(S5):
    for lo, hi, steep_mean, steep_sigma, flatinv_mean, flatinv_sigma in _ALPHA_BINS:
        if lo <= S5 < hi:
            return steep_mean, steep_sigma, flatinv_mean, flatinv_sigma

    # Below 0.1 Jy, use the first Tucci bin
    _, _, steep_mean, steep_sigma, flatinv_mean, flatinv_sigma = _ALPHA_BINS[0]
    return steep_mean, steep_sigma, flatinv_mean, flatinv_sigma


def _weights_from_pdf_grid(x, pdf):
    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    if x.ndim != 1 or pdf.shape != x.shape or x.size < 2:
        raise ValueError("PDF quadrature grids must be 1D arrays with at least two points.")

    dx = np.empty_like(x)
    dx[0] = 0.5 * (x[1] - x[0])
    dx[-1] = 0.5 * (x[-1] - x[-2])
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])

    w = pdf * dx
    total = np.sum(w)
    if total <= 0:
        raise ValueError("Zero integral in PDF grid.")

    return w / total


def _normal_grid(mean, sigma, ngrid, nsigma=6.0):
    x = np.linspace(mean - nsigma * sigma, mean + nsigma * sigma, ngrid)
    pdf = norm.pdf(x, loc=mean, scale=sigma)
    return x, _weights_from_pdf_grid(x, pdf)


def _truncnorm_grid(mean, sigma, low, high, ngrid):
    low_grid = mean - 8.0 * sigma if np.isneginf(low) else low
    high_grid = mean + 8.0 * sigma if np.isposinf(high) else high

    if high_grid <= low_grid:
        raise ValueError("Invalid truncnorm grid limits.")

    x = np.linspace(low_grid, high_grid, ngrid)
    a = (low - mean) / sigma
    b = (high - mean) / sigma
    pdf = truncnorm.pdf(x, a, b, loc=mean, scale=sigma)
    return x, _weights_from_pdf_grid(x, pdf)


def _prob_flat_given_flatinv(S5):
    """P(-0.5 <= alpha < 0.3 | alpha >= -0.5, S5)."""
    _, _, mu, sig = _alpha_bin_params(S5)
    z_flatinv = 1.0 - norm.cdf((-0.5 - mu) / sig)
    return (
        norm.cdf((0.3 - mu) / sig)
        - norm.cdf((-0.5 - mu) / sig)
    ) / z_flatinv


def _prob_inverted_given_flatinv(S5):
    return 1.0 - _prob_flat_given_flatinv(S5)


def _population_dNdlog10S5(S5, pop, dNdS_total_5GHz, dNdS_flatinv_5GHz):
    """Population-specific dN/dlog10(S5), in sr^-1."""
    S5 = np.asarray(S5, dtype=float)

    d_total = dNdS_total_5GHz(S5)
    d_flatinv = dNdS_flatinv_5GHz(S5)
    d_steep = np.clip(d_total - d_flatinv, 0.0, np.inf)

    if pop == "steep":
        dNdS = d_steep
    elif pop == "flatinv":
        dNdS = d_flatinv
    elif pop in ("flat", "inverted"):
        p_flat = np.array(
            [_prob_flat_given_flatinv(float(s)) for s in np.atleast_1d(S5)],
            dtype=float,
        ).reshape(S5.shape)
        dNdS = d_flatinv * (p_flat if pop == "flat" else 1.0 - p_flat)
    elif pop == "total":
        dNdS = d_total
    else:
        raise ValueError("Unknown population.")

    return np.log(10.0) * S5 * dNdS


def _populations_to_sum(pop):
    if pop in ("total", "all"):
        return ["steep", "flat", "inverted"]
    if pop == "flatinv":
        return ["flat", "inverted"]
    if pop in ("steep", "flat", "inverted"):
        return [pop]
    raise ValueError("Unknown population.")


def _flux_power_law(S5, nu, alpha):
    return S5 * (nu / 5.0) ** alpha


def _flux_power_law_array(S5, nu, alpha):
    nu = np.asarray(nu, dtype=float)
    return S5 * (nu / 5.0) ** alpha


def _flux_inverted_piecewise(S5, nu, nu_peak, alpha_before, alpha_after):
    if nu <= nu_peak:
        return _flux_power_law(S5, nu, alpha_before)

    S_peak = _flux_power_law(S5, nu_peak, alpha_before)
    return S_peak * (nu / nu_peak) ** alpha_after


def _flux_inverted_piecewise_array(
    S5, nu, nu_peak, alpha_before, alpha_after
):
    nu = np.asarray(nu, dtype=float)

    S_before = S5 * (nu / 5.0) ** alpha_before

    S_peak = S5 * (nu_peak / 5.0) ** alpha_before
    S_after = S_peak * (nu / nu_peak) ** alpha_after

    return np.where(nu <= nu_peak, S_before, S_after)


def _flux_broken_power_law(S5, nu, alpha_before, nu_break, alpha_after):
    if nu <= nu_break:
        return _flux_power_law(S5, nu, alpha_before)

    S_break = _flux_power_law(S5, nu_break, alpha_before)
    return S_break * (nu / nu_break) ** alpha_after


def _flux_broken_power_law_array(
    S5, nu, alpha_before, nu_break, alpha_after
):
    nu = np.asarray(nu, dtype=float)

    S_before = S5 * (nu / 5.0) ** alpha_before

    S_break = S5 * (nu_break / 5.0) ** alpha_before
    S_after = S_break * (nu / nu_break) ** alpha_after

    return np.where(nu <= nu_break, S_before, S_after)


def _nu_peak_grid_and_weights(n_nu_peak_per_bin):
    bins = np.array(
        [(5.0, 6.0), (6.0, 12.0), (12.0, 20.0), (20.0, 40.0), (40.0, 100.0)],
        dtype=float,
    )
    probs = np.array([0.27, 0.40, 0.15, 0.15, 0.03], dtype=float)
    probs /= probs.sum()

    values = []
    weights = []
    for (lo, hi), pbin in zip(bins, probs):
        nu = np.linspace(lo, hi, n_nu_peak_per_bin)
        w = np.ones_like(nu) * pbin / n_nu_peak_per_bin
        values.append(nu)
        weights.append(w)

    return np.concatenate(values), np.concatenate(weights)


def _alpha_after_peak_grid_and_weights(n_alpha_after):
    alpha0 = -0.10
    sigma = 0.53
    a = np.linspace(-2.2, 0.1, n_alpha_after)
    pdf = -(a - alpha0) * np.exp(-0.5 * ((a - alpha0) / sigma) ** 2)
    pdf = np.clip(pdf, 0.0, np.inf)
    return a, _weights_from_pdf_grid(a, pdf)


def _band_value(specification, nu_GHz, name):
    """Return a positive band-dependent scalar from scalar/mapping/callable input."""
    if callable(specification):
        value = specification(float(nu_GHz))
    elif np.isscalar(specification):
        value = specification
    else:
        try:
            value = specification[nu_GHz]
        except KeyError:
            matching_keys = [
                key for key in specification
                if np.isclose(float(key), float(nu_GHz))
            ]
            if not matching_keys:
                raise KeyError(f"No {name} value supplied for {nu_GHz} GHz.")
            value = specification[matching_keys[0]]

    value = float(value)
    if value <= 0:
        raise ValueError(f"All {name} values must be positive.")
    return value


class _TucciSpectralQuadrature:
    """
    Shared deterministic quadrature used by ncounts and cross-spectra
    as integrator for the expectation over the spectral parameters.
    """

    def __init__(
        self,
        n_alpha=80,
        n_delta=80,
        n_nu_peak_per_bin=8,
        n_alpha_after=80,
        flat_spectrum_model="C0",
    ):
        if flat_spectrum_model not in _ALLOWED_FLAT_MODELS:
            raise ValueError(
                f"flat_spectrum_model must be one of {sorted(_ALLOWED_FLAT_MODELS)}."
            )

        self.n_alpha = int(n_alpha)
        self.n_delta = int(n_delta)
        self.n_nu_peak_per_bin = int(n_nu_peak_per_bin)
        self.n_alpha_after = int(n_alpha_after)
        self.flat_spectrum_model = flat_spectrum_model

        # These distributions do not depend on S5, so computed once:
        # grid points (centers of \Delta x) and 
        # respective weights (\Delta x \times p(x)) from the pdf.
        
        # steep 
        self._steep_delta_grid, self._steep_delta_w = _normal_grid(
            mean=0.3, sigma=0.2, ngrid=self.n_delta
        )

        # inverted
        self._nu_peak_grid, self._nu_peak_w = _nu_peak_grid_and_weights(
            self.n_nu_peak_per_bin
        )
        self._alpha_after_grid, self._alpha_after_w = (
            _alpha_after_peak_grid_and_weights(self.n_alpha_after)
        )

    def iter_quadrature_points(self, S5, pop):
        """
        Yield (weight, flux_at_frequency) for the spectral-parameter
        quadrature points at fixed S5 and elementary population.
        """
        steep_mean, steep_sigma, flatinv_mean, flatinv_sigma = _alpha_bin_params(S5)

        if pop == "steep":
            alpha_grid, alpha_w = _truncnorm_grid(
                mean=steep_mean,
                sigma=steep_sigma,
                low=-np.inf,
                high=-0.5,
                ngrid=self.n_alpha,
            )

            for alpha, wa in zip(alpha_grid, alpha_w):
                for delta, wd in zip(self._steep_delta_grid, self._steep_delta_w):
                    alpha_high = alpha - delta
                    yield wa * wd, (
                        lambda nu, alpha_high=alpha_high: _flux_power_law(
                            S5, nu, alpha_high
                        )
                    )
            return

        if pop == "flat":
            alpha_grid, alpha_w = _truncnorm_grid(
                mean=flatinv_mean,
                sigma=flatinv_sigma,
                low=-0.5,
                high=0.3,
                ngrid=self.n_alpha,
            )

            for alpha, wa in zip(alpha_grid, alpha_w):
                if alpha < -0.1:
                    delta_mean, delta_sigma = 0.1, 0.1
                else:
                    delta_mean, delta_sigma = 0.2, 0.2

                delta_grid, delta_w = _normal_grid(
                    mean=delta_mean,
                    sigma=delta_sigma,
                    ngrid=self.n_delta,
                )

                for delta, wd in zip(delta_grid, delta_w):
                    alpha_fl = alpha - delta

                    if self.flat_spectrum_model == "C0":
                        yield wa * wd, (
                            lambda nu, alpha_fl=alpha_fl: _flux_power_law(
                                S5, nu, alpha_fl
                            )
                        )
                        continue

            return

        if pop == "inverted":
            alpha_grid, alpha_w = _truncnorm_grid(
                mean=flatinv_mean,
                sigma=flatinv_sigma,
                low=0.3,
                high=np.inf,
                ngrid=self.n_alpha,
            )

            for alpha, wa in zip(alpha_grid, alpha_w):
                for nu_peak, wp in zip(self._nu_peak_grid, self._nu_peak_w):
                    for alpha_after, wl in zip(self._alpha_after_grid, self._alpha_after_w):
                        yield wa * wp * wl, (
                            lambda nu,
                            alpha=alpha,
                            nu_peak=nu_peak,
                            alpha_after=alpha_after: _flux_inverted_piecewise(
                                S5,
                                nu,
                                nu_peak,
                                alpha,
                                alpha_after,
                            )
                        )
            return

        raise ValueError("pop must be 'steep', 'flat', or 'inverted'.")


    def iter_flux_vectors(self, S5, pop, nu_GHz):
        """
        Yield (weight, fluxes) for each spectral-parameter quadrature point, 
        where fluxes[i] = S_nu_i for all frequencies in nu_GHz.
        Thus the spectral quadrature is traversed only once for all
        requested frequencies.
        """
        nu_GHz = np.asarray(nu_GHz, dtype=float)

        steep_mean, steep_sigma, flatinv_mean, flatinv_sigma = (
            _alpha_bin_params(S5)
        )

        if pop == "steep":
            alpha_grid, alpha_w = _truncnorm_grid(
                mean=steep_mean,
                sigma=steep_sigma,
                low=-np.inf,
                high=-0.5,
                ngrid=self.n_alpha,
            )

            for alpha, wa in zip(alpha_grid, alpha_w):
                for delta, wd in zip(self._steep_delta_grid, self._steep_delta_w):
                    alpha_high = alpha - delta
                    fluxes = _flux_power_law_array(
                        S5,
                        nu_GHz,
                        alpha_high,
                    )
                    yield wa * wd, fluxes
            return

        if pop == "flat":
            alpha_grid, alpha_w = _truncnorm_grid(
                mean=flatinv_mean,
                sigma=flatinv_sigma,
                low=-0.5,
                high=0.3,
                ngrid=self.n_alpha,
            )

            for alpha, wa in zip(alpha_grid, alpha_w):
                if alpha < -0.1:
                    delta_mean, delta_sigma = 0.1, 0.1
                else:
                    delta_mean, delta_sigma = 0.2, 0.2

                delta_grid, delta_w = _normal_grid(
                    mean=delta_mean,
                    sigma=delta_sigma,
                    ngrid=self.n_delta,
                )

                for delta, wd in zip(delta_grid, delta_w):
                    alpha_fl = alpha - delta

                    if self.flat_spectrum_model == "C0":
                        fluxes = _flux_power_law_array(
                            S5,
                            nu_GHz,
                            alpha_fl,
                        )
                        yield wa * wd, fluxes
                        continue

            return

        if pop == "inverted":
            alpha_grid, alpha_w = _truncnorm_grid(
                mean=flatinv_mean,
                sigma=flatinv_sigma,
                low=0.3,
                high=np.inf,
                ngrid=self.n_alpha,
            )

            for alpha, wa in zip(alpha_grid, alpha_w):
                for nu_peak, wp in zip(self._nu_peak_grid, self._nu_peak_w):
                    for alpha_after, wl in zip(self._alpha_after_grid, self._alpha_after_w):
                        fluxes = _flux_inverted_piecewise_array(
                            S5,
                            nu_GHz,
                            nu_peak,
                            alpha,
                            alpha_after,
                        )
                        yield wa * wp * wl, fluxes
            return

        raise ValueError(
            "pop must be 'steep', 'flat', or 'inverted'."
        )


    def count_probability(self, S5, pop, nu_GHz, S_lim_jy):
        """
        P(S_nu > S_lim | S5, pop)
        """
        total = 0.0
        for weight, flux_at in self.iter_quadrature_points(S5, pop):
            if flux_at(nu_GHz) > S_lim_jy:
                total += weight
        return total

    def flux_product_expectation(
        self,
        S5,
        pop,
        nu1,
        nu2,
        flux_product_after_mask,
        apply_mask=False,
        mask_nu_GHz=None,
    ):
        """
        < S_nu1 S_nu2 M_12 | S5, pop >_theta_pop
        where M_12 is the mask factor (1 for unmasked, 0 for hard mask, or
        f_nu1 f_nu2 for Gaussian wings).
        """
        total = 0.0
        for weight, flux_at in self.iter_quadrature_points(S5, pop):
            S1 = flux_at(nu1)
            S2 = flux_at(nu2)
            Smask = flux_at(mask_nu_GHz) if apply_mask else None
            total += weight * flux_product_after_mask(S1, S2, Smask, nu1, nu2)
        return total

    def flux_moment_matrix(
        self,
        S5,
        pop,
        nu_GHz,
        apply_mask=False,
        mask_index=None,
        mask_Slim_jy=None,
        residual_flux_fraction=None,
    ):
        """
        Compute

            M_ij = <S_nu_i S_nu_j M_ij>

        simultaneously for all frequencies.

        Parameters
        ----------
        nu_GHz : array_like
            Frequencies at which fluxes are evaluated. If masking is
            enabled, this array must also contain mask_nu_GHz.

        mask_index : int or None
            Index of mask_nu_GHz in nu_GHz.

        residual_flux_fraction : array_like or None
            Fraction of source flux remaining after masking at each
            frequency. For a hard mask this can be all zeros.
        """
        nu_GHz = np.asarray(nu_GHz, dtype=float)

        nnu = len(nu_GHz)

        moment = np.zeros((nnu, nnu), dtype=float)

        if residual_flux_fraction is not None:
            residual_flux_fraction = np.asarray(
                residual_flux_fraction,
                dtype=float,
            )

        for weight, fluxes in self.iter_flux_vectors(
            S5,
            pop,
            nu_GHz,
        ):

            if apply_mask:
                Smask = fluxes[mask_index]

                if Smask >= mask_Slim_jy:
                    if residual_flux_fraction is None:
                        # hard mask
                        continue

                    fluxes_eff = (
                        fluxes * residual_flux_fraction
                    )
                else:
                    fluxes_eff = fluxes
            else:
                fluxes_eff = fluxes

            moment += weight * np.outer(
                fluxes_eff,
                fluxes_eff,
            )

        return moment


def _validate_S5_grid(S5_grid_jy):
    S5_grid_jy = np.asarray(S5_grid_jy, dtype=float)
    if np.any(S5_grid_jy <= 0):
        raise ValueError("S5_grid_jy must be strictly positive.")

    logS5_grid = np.log10(S5_grid_jy)
    if np.any(np.diff(logS5_grid) <= 0):
        raise ValueError("S5_grid_jy must be strictly increasing.")

    return S5_grid_jy, logS5_grid


def _validate_population(population):
    if population not in _ALLOWED_POPULATIONS:
        raise ValueError(f"population must be one of {sorted(_ALLOWED_POPULATIONS)}.")



def tucci_cross_spectra(
    nu_pairs_GHz,
    S5_grid_jy,
    dNdS_total_5GHz,
    dNdS_flatinv_5GHz,
    mask_nu_GHz=None,
    mask_Slim_jy=None,
    mask_beam_fwhm_arcmin=None,
    point_source_fwhm_arcmin=None,
    mask_f_fwhm=1.0,
    population="total",
    n_alpha=80,
    n_delta=80,
    n_nu_peak_per_bin=8,
    n_alpha_after=80,
    unit="uK2",
    pol_frac=0.028,
    flat_spectrum_model="C0",
    verbose=False
):
    """
    Analytical Tucci radio-source Poisson BB cross-spectra.

    The implementation evaluates, for each elementary population t,

        int dlog10(S5) [dN_t/dlog10(S5)]
            < S_nu1 S_nu2 M_12 >_{theta_t | S5,t},

    where M_12=1 for an unmasked source. For a source selected as masked,
    M_12=0 for a hard mask, or M_12=f_nu1 f_nu2 when Gaussian wings are
    retained. The final intensity shot-noise spectrum is multiplied by
    pol_frac**2/2 to obtain BB.

    Parameters
    ----------
    nu_pairs_GHz : iterable of tuple
        Frequency pairs (nu1, nu2), in GHz, for which the auto- or
        cross-spectra are computed.

    S5_grid_jy : array_like
        Positive, strictly increasing grid of 5 GHz flux densities in Jy
        over which the number-count integral is evaluated.

    dNdS_total_5GHz : callable
        Total differential number counts at 5 GHz. It must accept S5 in Jy
        and return dN/dS5 in Jy^-1 sr^-1.

    dNdS_flatinv_5GHz : callable
        Differential number counts of the combined flat+inverted population
        at 5 GHz. It must accept S5 in Jy and return dN/dS5 in Jy^-1 sr^-1.

    mask_nu_GHz : float or None, optional
        Reference frequency in GHz at which sources to be masked are selected.
        Sources with S_nu >= mask_Slim_jy are treated as masked. If None,
        no source masking is applied. Default is None.

    mask_Slim_jy : float or None, optional
        Flux-density threshold in Jy used to select masked sources at
        mask_nu_GHz. It must be provided together with mask_nu_GHz.
        Default is None.

    mask_beam_fwhm_arcmin : float, mapping, callable, or None, optional
        FWHM in arcmin used to define the circular mask radius. It may be a
        scalar, a mapping keyed by frequency, or a callable of frequency.
        If omitted together with point_source_fwhm_arcmin, masked sources are
        completely removed. Default is None.

    point_source_fwhm_arcmin : float, mapping, callable, or None, optional
        Gaussian FWHM in arcmin of the observed point-source profile in each
        frequency band. It may be a scalar, a mapping keyed by frequency, or
        a callable of frequency. Required together with
        mask_beam_fwhm_arcmin when Gaussian wings are retained.
        Default is None.

    mask_f_fwhm : float, optional
        Fraction of mask_beam_fwhm_arcmin used to mask, so that
        R_mask = mask_f_fwhm * mask_beam_fwhm_arcmin.
        Default is 1.0.

    population : str, optional
        Source population to include. One of 'total', 'steep', 'flat',
        'inverted', 'flatinv', or 'all'. Default is 'total'.

    n_alpha : int, optional
        Number of quadrature points used for the 1--5 GHz spectral index
        alpha_1^5. Default is 80.

    n_delta : int, optional
        Number of quadrature points used for the high-frequency spectral
        steepening Delta alpha of steep- and flat-spectrum sources.
        Default is 80.

    n_nu_peak_per_bin : int, optional
        Number of quadrature points used within each peak-frequency bin for
        inverted-spectrum sources. Default is 8.

    n_alpha_after : int, optional
        Number of quadrature points used for the spectral index after the
        peak of inverted-spectrum sources. Default is 80.

    unit : str, optional
        Output unit. Either 'uK2' for thermodynamic CMB temperature units
        or 'Jy2sr' for intensity units in Jy^2 sr^-1. Default is 'uK2'.

    pol_frac : float, optional
        Effective polarization fraction used to convert the intensity
        shot-noise spectrum into BB through the factor pol_frac**2 / 2.
        Default is 0.028.

    flat_spectrum_model : str, optional
        High-frequency spectral prescription for flat-spectrum sources.
        Model already implemented 'C0'.

    Returns
    -------
    dict
        If population is not 'all', returns a dictionary of the form
        {(nu1, nu2): C_l_BB} for the requested population.

        If population is 'all', returns a nested dictionary containing
        the spectra for 'total', 'steep', 'flat', 'inverted', and 'flatinv'.
    """
    S5_grid_jy, logS5_grid = _validate_S5_grid(S5_grid_jy)
    _validate_population(population)

    if unit not in {"uK2", "Jy2sr"}:
        raise ValueError("unit must be 'uK2' or 'Jy2sr'.")

    if (mask_nu_GHz is None) != (mask_Slim_jy is None):
        raise ValueError("Provide both mask_nu_GHz and mask_Slim_jy, or neither.")

    apply_mask = mask_nu_GHz is not None
    partial_mask = (
        mask_beam_fwhm_arcmin is not None
        or point_source_fwhm_arcmin is not None
    )

    if partial_mask and not apply_mask:
        raise ValueError(
            "Beam-dependent masking requires mask_nu_GHz and mask_Slim_jy."
        )

    if (mask_beam_fwhm_arcmin is None) != (point_source_fwhm_arcmin is None):
        raise ValueError(
            "Provide both mask_beam_fwhm_arcmin and point_source_fwhm_arcmin, or neither."
        )

    mask_f_fwhm = float(mask_f_fwhm)
    if mask_f_fwhm <= 0:
        raise ValueError("mask_f_fwhm must be positive.")

    def residual_flux_fraction(nu_GHz):
        """
        Fraction of the integrated flux remaining outside the mask at frequency nu_GHz.
        """
        if not partial_mask:
            return 0.0

        mask_fwhm = _band_value(
            mask_beam_fwhm_arcmin, nu_GHz, "mask_beam_fwhm_arcmin"
        )
        source_fwhm = _band_value(
            point_source_fwhm_arcmin, nu_GHz, "point_source_fwhm_arcmin"
        )
        mask_radius = mask_f_fwhm * mask_fwhm / 2.0
        return float(
            np.exp(-4.0 * np.log(2.0) * (mask_radius / source_fwhm) ** 2)
        )

    def flux_product_after_mask(S1, S2, Smask, nu1, nu2):
        if not apply_mask or Smask < mask_Slim_jy:
            return S1 * S2

        f1 = residual_flux_fraction(nu1)
        f2 = residual_flux_fraction(nu2)
        return f1 * f2 * S1 * S2

    print("Initializing Tucci spectral quadrature...") if verbose else None
    quadrature = _TucciSpectralQuadrature(
        n_alpha=n_alpha,
        n_delta=n_delta,
        n_nu_peak_per_bin=n_nu_peak_per_bin,
        n_alpha_after=n_alpha_after,
        flat_spectrum_model=flat_spectrum_model
    )

    nu_pairs = [(float(a), float(b)) for a, b in nu_pairs_GHz]
    requested_nu = np.unique(np.asarray([nu for pair in nu_pairs for nu in pair], dtype=float))
    if apply_mask:
        eval_nu = np.unique(
            np.append(requested_nu, float(mask_nu_GHz))
        )
    else:
        eval_nu = requested_nu.copy()

    def frequency_index(nu):
        idx = np.where(np.isclose(eval_nu, float(nu)))[0]

        if len(idx) != 1:
            raise RuntimeError(f"Could not uniquely locate frequency {nu}.")

        return int(idx[0])

    pair_indices = {
        pair: (
            frequency_index(pair[0]),
            frequency_index(pair[1]),
        )
        for pair in nu_pairs
    }

    mask_index = (
        frequency_index(mask_nu_GHz)
        if apply_mask
        else None
    )

    # Counts only depend on S5 and population, so compute them once instead
    # of once per frequency pair.
    elementary_pops = _populations_to_sum(population)
    dNdlog_by_pop = {
        t: _population_dNdlog10S5(
            S5_grid_jy, t, dNdS_total_5GHz, dNdS_flatinv_5GHz
        )
        for t in elementary_pops
    }

    # Compute the residual flux fraction at each frequency
    # (if masking is applied and Gaussian wings are retained)
    if apply_mask and partial_mask:
        residual_fraction = np.array(
            [
                residual_flux_fraction(nu)
                for nu in eval_nu
            ],
            dtype=float,
        )
    else:
        residual_fraction = None

    def compute_for_population(pop):
        print(f"\nComputing cross-spectra for population '{pop}'...") if verbose else None
        formula_pops = _populations_to_sum(pop)

        local_dNdlog = {
            t: (
                dNdlog_by_pop[t]
                if t in dNdlog_by_pop
                else _population_dNdlog10S5(
                    S5_grid_jy,
                    t,
                    dNdS_total_5GHz,
                    dNdS_flatinv_5GHz,
                )
            )
            for t in formula_pops
        }

        nnu = len(eval_nu)

        integrand_matrix = np.zeros(
            (len(S5_grid_jy), nnu, nnu),
            dtype=float,
        )

        for t in formula_pops:

            dNdlog = local_dNdlog[t]

            for iS, S5 in enumerate(S5_grid_jy):
                print(f"  Population '{t}', S5={S5:.3e} Jy ({iS+1}/{len(S5_grid_jy)})") if verbose else None

                moment = quadrature.flux_moment_matrix(
                    S5=float(S5),
                    pop=t,
                    nu_GHz=eval_nu,
                    apply_mask=apply_mask,
                    mask_index=mask_index,
                    mask_Slim_jy=mask_Slim_jy,
                    residual_flux_fraction=residual_fraction,
                )

                integrand_matrix[iS] += (
                    dNdlog[iS] * moment
                )

        cl_matrix = np.trapezoid(
            integrand_matrix,
            logS5_grid,
            axis=0,
        )

        cl_matrix *= pol_frac**2 / 2.0

        result = {}

        for pair in nu_pairs:
            nu1, nu2 = pair

            i, j = pair_indices[pair]

            cl = float(cl_matrix[i, j])

            if unit == "uK2":
                cl *= jysr2cmb(nu1) * jysr2cmb(nu2) * 1e12
            result[pair] = cl

        return result

    if population == "all":
        steep = compute_for_population("steep")
        flat = compute_for_population("flat")
        inverted = compute_for_population("inverted")

        total = {pair: steep[pair] + flat[pair] + inverted[pair] for pair in nu_pairs}
        flatinv = {pair: flat[pair] + inverted[pair] for pair in nu_pairs}

        return {
            "total": total,
            "steep": steep,
            "flat": flat,
            "inverted": inverted,
            "flatinv": flatinv,
        }

    return compute_for_population(population)



def tucci_number_counts(
    nu_GHz,
    S_lim_jy,
    S5_grid_jy,
    dNdS_total_5GHz,
    dNdS_flatinv_5GHz,
    population="total",
    n_alpha=80,
    n_delta=80,
    n_nu_peak_per_bin=8,
    flat_spectrum_model="C0",
    n_alpha_after=80
):
    """
    Integral number counts N(>S_lim) per sr at one observing frequency.

    Evaluates

        N_nu(>S_lim)
        =
        sum_t int dlog10(S5) [dN_t/dlog10(S5)]
            < 1[S_nu(S5, theta_t, t) > S_lim] >_{theta_t},

    where the expectation value over the spectral parameters is evaluated
    by deterministic numerical quadrature.

    The same population split, spectral prescriptions and quadrature as
    tucci_cross_spectra are used.

    Parameters
    ----------
    nu_GHz : float
        Observing frequency in GHz.

    S_lim_jy : float
        Flux-density threshold in Jy. Sources with S_nu > S_lim_jy are
        included in the integral number counts.

    S5_grid_jy : array_like
        Positive, strictly increasing grid of 5 GHz flux densities in Jy
        over which the number-count integral is evaluated.

    dNdS_total_5GHz : callable
        Total differential number counts at 5 GHz. It must accept S5 in Jy
        and return dN/dS5 in Jy^-1 sr^-1.

    dNdS_flatinv_5GHz : callable
        Differential number counts of the combined flat+inverted population
        at 5 GHz. It must accept S5 in Jy and return dN/dS5 in Jy^-1 sr^-1.

    population : str, optional
        Source population to include. One of 'total', 'steep', 'flat',
        'inverted', 'flatinv', or 'all'. Default is 'total'.

    n_alpha : int, optional
        Number of quadrature points used for the 1--5 GHz spectral index
        alpha_1^5. Default is 80.

    n_delta : int, optional
        Number of quadrature points used for the high-frequency spectral
        steepening Delta alpha of steep and flat sources. Default is 80.

    n_nu_peak_per_bin : int, optional
        Number of quadrature points used within each peak-frequency bin for
        inverted-spectrum sources. Default is 8.

    flat_spectrum_model : str, optional
        High-frequency spectral prescription for flat-spectrum sources.
        Model already implemented 'C0'.

    n_alpha_after : int, optional
        Number of quadrature points used for the spectral index after the
        peak of inverted-spectrum sources. Default is 80.

    Returns
    -------
    float or dict
        Integral number counts in sr^-1. If population is 'all', returns a
        dictionary containing the counts for 'total', 'steep', 'flat',
        'inverted', and 'flatinv'; otherwise returns the counts for the
        requested population.
    """
    S5_grid_jy, logS5_grid = _validate_S5_grid(S5_grid_jy)
    _validate_population(population)

    nu_GHz = float(nu_GHz)
    S_lim_jy = float(S_lim_jy)

    quadrature = _TucciSpectralQuadrature(
        n_alpha=n_alpha,
        n_delta=n_delta,
        n_nu_peak_per_bin=n_nu_peak_per_bin,
        n_alpha_after=n_alpha_after,
        flat_spectrum_model=flat_spectrum_model
    )

    def compute_for_population(pop):
        integrand_total = np.zeros_like(S5_grid_jy, dtype=float)

        for t in _populations_to_sum(pop):
            dNdlogS5 = _population_dNdlog10S5(
                S5_grid_jy, t, dNdS_total_5GHz, dNdS_flatinv_5GHz
            )
            inner = np.array(
                [
                    quadrature.count_probability(
                        S5=float(S5),
                        pop=t,
                        nu_GHz=nu_GHz,
                        S_lim_jy=S_lim_jy,
                    )
                    for S5 in S5_grid_jy
                ]
            )
            integrand_total += dNdlogS5 * inner

        return float(np.trapezoid(integrand_total, logS5_grid))

    if population == "all":
        steep = compute_for_population("steep")
        flat = compute_for_population("flat")
        inverted = compute_for_population("inverted")
        return {
            "steep": steep,
            "flat": flat,
            "inverted": inverted,
            "flatinv": flat + inverted,
            "total": steep + flat + inverted,
        }

    return compute_for_population(population)

