"""Read semileptonic top parquet files into flat token arrays, vectorised over events."""

import awkward as ak
import numpy as np
import torch

from experiments.hep import PtPhiEtaM_to_EPPP, lorentz_boost

# tokens of an event: [tophad, b subjet, harder subjet, softer subjet, constituents, tail]
# with the non-b subjets in decreasing energy in the hadronic top rest frame (or pt), the constituents
# grouped by subjet in that order
# and pt-sorted within a group, and the tail holding the objects outside the hadronic top
N_HEAD = 4
N_SUBJETS = 3
TAIL = ("TopLep", "Lepton")

B_TAGS = (-5, 5)
DOWN_TAGS = (-3, -1, 1, 3)

# [charge, is_electron, is_muon, is_photon, is_charged_hadron, is_neutral_hadron] per PDG id,
# the last row is the fallback for ids outside the table
PID_CODES = np.array([22, 11, -11, 13, -13, 211, -211, 321, -321, 130, 2212, -2212, 2112, -2112])
PID_FEATURES = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [-1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [-1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [-1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [-1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [-1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float64,
)
PHOTON_PID = int(np.nonzero(PID_CODES == 22)[0][0])


def _fourmomenta(record):
    keys = ["pt", "phi", "eta"]
    columns = [np.asarray(ak.to_numpy(record[key]), dtype=np.float64) for key in keys]
    # the lepton is stored with its charge instead of a mass
    columns.append(
        np.asarray(ak.to_numpy(record["mass"]), dtype=np.float64)
        if "mass" in record.fields
        else np.zeros_like(columns[0])
    )
    return PtPhiEtaM_to_EPPP(torch.from_numpy(np.stack(columns, axis=-1))).numpy()


def rest_frame_energy(fourmomenta, tophad):
    """Energy of fourmomenta (..., 4) in the rest frame of the hadronic tops tophad (n, 4)."""
    shape = fourmomenta.shape[:-1]
    frame = tophad.reshape(shape[0], *[1] * (len(shape) - 1), 4).expand(*shape, 4)
    boosted = lorentz_boost(fourmomenta.reshape(-1, 4), frame.reshape(-1, 4))
    return boosted[:, 0].reshape(shape)


def _subjet_order(tags, hardness):
    """Return the subjet indices in (b, harder non-b, softer non-b) order."""
    is_b = np.isin(tags, B_TAGS)
    assert (is_b.sum(1) == 1).all(), "every event needs exactly one b-tagged subjet"
    # stable, so ties keep the order of the file
    non_b = np.argsort(np.where(is_b, np.inf, -hardness), axis=1, kind="stable")[:, :2]
    return np.concatenate([np.argmax(is_b, axis=1)[:, None], non_b], axis=1)


def read_events(parquet_path, tail=TAIL, pt_order=False):
    """Return the tokens of all events of one parquet file as flat arrays, the non-b subjets
    ordered by rest frame energy, or by pt with pt_order.

    x: fourmomenta (E, px, py, pz) of all tokens of all events, concatenated
    pid: row of PID_FEATURES per token, -1 for tokens without a PID
    nconst: constituents per subjet, in token order
    ntokens: tokens per event
    tags: parton tag per subjet, in token order
    lep_charge: charge of the lepton
    """
    events = ak.from_parquet(parquet_path)
    subjets = events["Subjets"]
    assert (ak.to_numpy(ak.num(subjets, axis=1)) == N_SUBJETS).all(), (
        f"{parquet_path} has events with more or less than {N_SUBJETS} subjets"
    )

    tophad = _fourmomenta(events["TopHad"])
    if pt_order:
        hardness = np.asarray(ak.to_numpy(subjets["pt"]), dtype=np.float64)
    else:
        hardness = rest_frame_energy(
            torch.from_numpy(_fourmomenta(subjets)), torch.from_numpy(tophad)
        ).numpy()
    order = _subjet_order(ak.to_numpy(subjets["tag"]), hardness)
    subjets = subjets[np.arange(len(events))[:, None], ak.Array(order)]

    constituents = subjets["Constituents"]
    constituents = constituents[ak.argsort(constituents["pt"], axis=2, ascending=False)]
    nconst = ak.to_numpy(ak.num(constituents, axis=2)).astype(np.int32)
    constituents = ak.flatten(ak.flatten(constituents, axis=2), axis=1)

    ntokens = N_HEAD + nconst.sum(1) + len(tail)
    offsets = np.concatenate([[0], np.cumsum(ntokens)])
    const_offsets = np.concatenate([[0], np.cumsum(nconst.sum(1))])
    const_tokens = (
        offsets[:-1].repeat(nconst.sum(1))
        + N_HEAD
        + (np.arange(const_offsets[-1]) - const_offsets[:-1].repeat(nconst.sum(1)))
    )

    x = np.zeros((offsets[-1], 4), dtype=np.float64)
    x[offsets[:-1]] = tophad
    x[offsets[:-1, None] + 1 + np.arange(N_SUBJETS)] = _fourmomenta(subjets)
    x[const_tokens] = _fourmomenta(constituents)
    for i, name in enumerate(tail):
        x[offsets[1:] - len(tail) + i] = _fourmomenta(events[name])

    pid = np.full(offsets[-1], -1, dtype=np.int8)
    const_pid = ak.to_numpy(constituents["pid"]).astype(np.int64)
    codes = np.argsort(PID_CODES)
    matches = np.searchsorted(PID_CODES[codes], const_pid).clip(max=len(PID_CODES) - 1)
    matches = np.where(PID_CODES[codes][matches] == const_pid, codes[matches], len(PID_CODES))
    pid[const_tokens] = matches

    return {
        "x": x,
        "pid": pid,
        "nconst": nconst,
        "ntokens": ntokens,
        "tags": ak.to_numpy(subjets["tag"]).astype(np.int8),
        "lep_charge": np.asarray(ak.to_numpy(events["Lepton"]["charge"]), dtype=np.float32),
    }
