""""
Function to compute the fsky loss from the original mask due to PS masks.
"""
import numpy as np



def compute_fsky_PS_mask(fsky_init_binary, fsky_init_weighted, N_PS, FWHM, f_fwhm):
    """
    Compute the fsky loss from the original mask due to PS masks.
    
    Parameters
    ----------
    fsky_init_binary : float
        The initial fsky of the binary mask before applying the PS masks.
    fsky_init_weighted : float
        The initial fsky of the weighted and apodized mask before applying the PS masks.
    N_PS : int
        The number of point sources to mask.
    FWHM : float
        The full width at half maximum (FWHM) of the PS mask in arcminutes.
    f_fwhm : float
        The fraction of the FWHM to use for the PS mask diameter.
    
    Returns
    -------
    area_masked : float
        The area masked by the PS masks in steradians.
    fsky_loss : float
        The loss in fsky due to the PS masks.
    """
    # Area initial mask in steradians
    area_init_binary = fsky_init_binary * 4 * np.pi
    area_init_weighted = fsky_init_weighted * 4 * np.pi

    # Area of a single PS mask in steradians
    FWHM_rad = np.radians(FWHM / 60.0)
    radius_rad = f_fwhm * FWHM_rad / 2.
    area_PS_mask = np.pi * radius_rad**2
    
    # Prob that a point of the initial fsky is masked by at least one PS mask
    prob_masked = 1 - (np.exp(-N_PS * area_PS_mask / (area_init_binary)))

    # Area masked by PS masks
    area_masked = prob_masked * area_init_weighted
    fsky_loss = area_masked / (4 * np.pi)

    # Percentage of fsky lost relative to the initial weighted fsky
    percentage_of_fsky_lost = (fsky_loss / fsky_init_weighted)

    # fsky remaining after applying PS masks
    fsky_remaining_after_PS_masks = fsky_init_weighted - fsky_loss
    
    return area_masked, fsky_loss, percentage_of_fsky_lost, fsky_remaining_after_PS_masks
