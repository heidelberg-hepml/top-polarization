"""Match reconstructed subjets to truth partons or to Delphes b-jets."""

from __future__ import annotations

from .config import ProcessConfig
from .kinematics import deltaR


def match_parton_jets(
    partons,
    subjets,
    process: ProcessConfig,
    max_dr: float = 0.4,
):
    """Assign (b, u-type, d-type) tags to subjets via deltaR to truth partons.

    Returns the matched subjets with `user_index` set to the signed quark id,
    ordered with the b-candidate first. Returns None if matches collide or any
    deltaR exceeds `max_dr`.
    """
    ids = [p.id for p in partons]
    test_id = ids[process.decay_test_index]
    if test_id in [11, -11, 13, -13]:
        hadronic = "top"
    elif test_id in [1, -1, 2, -2, 3, -3, 4, -4]:
        hadronic = "atop"
    else:
        raise ValueError(
            f"Parton-level objects do not match expected IDs. "
            f"The ID of the lepton/quark is {test_id}."
        )

    b_hadronic = (
        partons[process.b_index_when_top_hadronic]
        if hadronic == "top"
        else partons[process.b_index_when_atop_hadronic]
    )

    u_index = next((i for i, p in enumerate(partons) if abs(p.id) in [2, 4]), None)
    d_index = next((i for i, p in enumerate(partons) if abs(p.id) in [1, 3]), None)
    if u_index is None or d_index is None:
        raise ValueError(
            "Could not find both up-type and down-type quarks in parton-level objects."
        )
    u_hadronic, d_hadronic = partons[u_index], partons[d_index]

    from pylorentz import Momentum4  # heavy dep; only needed in `rootpy` env

    matches = []
    for tag, ps, uidx in zip(
        ["b", "u", "d"],
        [b_hadronic, u_hadronic, d_hadronic],
        [5, u_hadronic.id, d_hadronic.id],
        strict=True,
    ):
        drs = [deltaR(sj, Momentum4(ps.e, ps.px, ps.py, ps.pz)) for sj in subjets]
        min_dr = min(drs)
        min_idx = drs.index(min_dr)
        subjets[min_idx].set_user_index(uidx)
        matches.append((tag, min_idx, min_dr))

    indices = [m[1] for m in matches]
    if len(indices) != len(set(indices)):
        return None
    if any(m[2] > max_dr for m in matches):
        return None

    b_cand = subjets[matches[0][1]]
    matched = [sj for c, sj in enumerate(subjets) if c in indices]
    return [b_cand] + [sj for sj in matched if sj != b_cand]


def match_delphes_b_jet(b_tagged_jets, subjets, max_dr: float = 0.1):
    """Find the subjet closest to a Delphes b-tagged jet; tag it as b."""
    min_dr = 1e6
    sj_cand = None
    for sj in subjets:
        for b in b_tagged_jets:
            dr = deltaR(sj, b)
            if dr < min_dr:
                min_dr = dr
                sj_cand = sj
    if min_dr <= max_dr:
        return [sj_cand] + [sj for sj in subjets if sj != sj_cand]
    return None
