"""Lightweight kinematic helpers used across the pipeline.

All functions are pure: no shared state. The phi/eta accessors handle both
FastJet PseudoJet (callable methods) and Delphes objects (attributes).
"""

from __future__ import annotations

import math


def phi(p) -> float:
    if callable(getattr(p, "phi", None)):
        return p.phi()
    return p.phi if hasattr(p, "phi") else p.Phi


def eta(p) -> float:
    if callable(getattr(p, "eta", None)):
        return p.eta()
    return p.eta if hasattr(p, "eta") else p.Eta


def deltaR(p1, p2) -> float:
    dphi = abs(phi(p1) - phi(p2))
    deta = eta(p1) - eta(p2)
    if dphi > math.pi:
        dphi = dphi - 2 * math.pi
    return math.sqrt(deta**2 + dphi**2)


def inv_mass(subjets) -> float:
    s = subjets[0]
    for j in subjets[1:]:
        s = s + j
    return s.m()


def is_in_jet_radius(particle, jet, R: float = 1.5) -> bool:
    return deltaR(particle, jet) <= R


def delphes_to_fj(obj):
    """Convert a Delphes object with (PT, Phi, Eta) into a fastjet PseudoJet."""
    import fastjet as fj

    px = obj.PT * math.cos(obj.Phi)
    py = obj.PT * math.sin(obj.Phi)
    pz = obj.PT * math.sinh(obj.Eta)
    E = obj.PT * math.cosh(obj.Eta)
    return fj.PseudoJet(px, py, pz, E)


def fj_parents(cluster, jet):
    """Return the two FastJet parents of `jet` in `cluster`, or (None, None)."""
    import fastjet as fj

    p1 = fj.PseudoJet()
    p2 = fj.PseudoJet()
    if not cluster.has_parents(jet, p1, p2):
        return None, None
    return p1, p2
