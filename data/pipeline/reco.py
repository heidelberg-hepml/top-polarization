"""Top-quark reconstruction (hadronic and leptonic side)."""

from __future__ import annotations

import itertools
import math

from .kinematics import delphes_to_fj, inv_mass


def reco_hadronic_top(subjets, top_mass_range, mtop: int = 173):
    """Pick the (3-jet or 4-jet) subset closest to mtop, within the mass window.

    Returns the chosen subjets, or None if the best candidate falls outside
    `top_mass_range = (m_lo, m_hi)`.
    """
    best_subjets = subjets
    m = inv_mass(subjets)
    best_diff = abs(m - mtop)
    best_mass = m

    for candidates in itertools.combinations(subjets, 3):
        m = inv_mass(candidates)
        diff = abs(m - mtop)
        if diff < best_diff:
            best_diff = diff
            best_mass = m
            best_subjets = candidates

    if best_mass < int(top_mass_range[0]) or best_mass > int(top_mass_range[1]):
        return None
    return best_subjets


def reco_w(lepton, neutrino, w_mass: float = 80):
    """Reconstruct the W from (lepton, MET) using the W-mass constraint to fix pz."""
    import fastjet as fj

    pxnu = neutrino.MET * math.cos(neutrino.Phi)
    pynu = neutrino.MET * math.sin(neutrino.Phi)

    coeff = w_mass**2 / 2 + lepton.px() * pxnu + lepton.py() * pynu
    transverse_cond = coeff**2 - (lepton.pt()) ** 2 * (neutrino.MET) ** 2

    if transverse_cond < 0:
        met_corrected = coeff / lepton.pt()
        pxnu = met_corrected * math.cos(neutrino.Phi)
        pynu = met_corrected * math.sin(neutrino.Phi)
        pznu = coeff * lepton.pz() / (lepton.pt()) ** 2
        enu = math.sqrt(met_corrected**2 + pznu**2)
    else:
        pznu1 = coeff * lepton.pz() / (lepton.pt()) ** 2 + (
            lepton.e() / (lepton.pt()) ** 2
        ) * math.sqrt(transverse_cond)
        pznu2 = coeff * lepton.pz() / (lepton.pt()) ** 2 - (
            lepton.e() / (lepton.pt()) ** 2
        ) * math.sqrt(transverse_cond)
        pznu1, pznu2 = pznu1.real, pznu2.real
        pznu = pznu1 if abs(pznu1) <= abs(pznu2) else pznu2
        enu = math.sqrt(neutrino.MET**2 + pznu**2)

    nu = fj.PseudoJet(pxnu, pynu, pznu, enu)
    return lepton + nu


def reco_leptonic_top(w, lepton, b_quark, top_mass_range):
    """Combine W + b-jet; if outside window, try (W + b - lepton). Else None."""
    b = delphes_to_fj(b_quark)
    t1 = w + b
    if t1.m() < int(top_mass_range[0]) or t1.m() > int(top_mass_range[1]):
        t2 = w + b - lepton
        if t2.m() < int(top_mass_range[0]) or t2.m() > int(top_mass_range[1]):
            return None
        return t2
    return t1
