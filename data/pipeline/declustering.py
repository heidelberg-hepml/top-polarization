"""FatJet declustering strategies.

- "pnet": follow https://arxiv.org/pdf/2407.01663 — peel off the heavier
  parent until its mass falls below m_max.
- "ca":   follow https://cds.cern.ch/record/1194489/files/JME-09-001-pas.pdf
  — peel off softer branches below a relative pT threshold.
"""

from __future__ import annotations

from .kinematics import fj_parents


def decluster(
    constits, strategy: str = "pnet", m_max: float = 30, threshold: float = 0.05, R: float = 1.5
):
    """Decluster a list of Delphes constituents into <=4 subjets.

    Returns (cluster, subjets) — the caller must hold onto `cluster` until
    it's done reading `.constituents()` etc. off the subjets. Returns
    (cluster, None) if declustering produces 2 or fewer subjets.

    Background: fastjet_cxx's delete_self_when_unused() does not propagate
    the C++ ownership semantics back through the SWIG binding, so the only
    reliable way to keep the ClusterSequence alive is to hold a Python
    reference.
    """
    import fastjet as fj

    jet_def = fj.JetDefinition(fj.cambridge_algorithm, R)
    fj_particles = []
    for constit in constits:
        charge = constit.Charge if hasattr(constit, "Charge") else 0
        pid = constit.PID if hasattr(constit, "PID") else 0
        p4 = constit.P4()
        part = fj.PseudoJet(p4.Px(), p4.Py(), p4.Pz(), p4.E())
        part.set_user_index(int(pid * 3 + (charge + 1)))
        fj_particles.append(part)

    cluster = fj.ClusterSequence(fj_particles, jet_def)
    jet_fj = max(cluster.inclusive_jets(), key=lambda j: j.pt())

    if strategy == "pnet":
        subjets = _rec_pnet(cluster, jet_fj, m_max=m_max)
    elif strategy == "ca":
        subjets = _rec_ca(cluster, jet_fj, pt_hard=jet_fj.pt(), threshold=threshold)
    else:
        raise ValueError(f"Unknown declustering strategy: {strategy}")

    if len(subjets) <= 2:
        return cluster, None
    return cluster, sorted(subjets, key=lambda s: s.pt(), reverse=True)[:4]


def _rec_pnet(cluster, child, m_max: float):
    parent1, parent2 = fj_parents(cluster, child)
    if parent1 is None or parent2 is None:
        return [child]
    if parent2.m() > parent1.m():
        parent1, parent2 = parent2, parent1
    if parent1.m() < m_max:
        return [parent1, parent2]
    if parent1.m() > m_max and parent2.m() < m_max:
        return _rec_pnet(cluster, parent1, m_max) + [parent2]
    if parent2.m() > m_max:
        return _rec_pnet(cluster, parent1, m_max) + _rec_pnet(cluster, parent2, m_max)
    return [child]


def _rec_ca(cluster, child, pt_hard: float, threshold: float, depth: int = 0):
    parent1, parent2 = fj_parents(cluster, child)
    if parent1 is None or parent2 is None:
        return [child]
    if parent2.pt() > parent1.pt():
        parent1, parent2 = parent2, parent1
    if parent1.pt() < threshold * pt_hard:
        return [child]
    if parent1.pt() > threshold * pt_hard and parent2.pt() < threshold * pt_hard:
        return _rec_ca(cluster, parent1, pt_hard, threshold, depth)
    if parent2.pt() > threshold * pt_hard:
        if depth == 0:
            return _rec_ca(cluster, parent1, pt_hard, threshold, 1) + _rec_ca(
                cluster, parent2, pt_hard, threshold, 1
            )
        if depth == 1:
            return [parent1, parent2]
    return [child]
