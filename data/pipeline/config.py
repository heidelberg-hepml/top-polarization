"""Per-process configuration for the MC pipeline.

A ProcessConfig captures the bits that differ between ttbar and ttH (and
future processes): which extra event-level cuts apply, how the LHE parton
list is indexed for the b-quark on each top, and which extra records get
written to the output parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ProcessConfig:
    name: str  # "tt" | "tth" | "tt_ctg" | "tt_ctg_lin"

    # Index into `event.particles[2:]` that identifies whether the LEPTONIC
    # top is "top" or "atop" (negative indices into the parton list).
    decay_test_index: int

    # Index of the b-quark from the hadronic top, when the hadronic top is
    # the "top" (positive) and when it is the "atop" side, respectively.
    b_index_when_top_hadronic: int
    b_index_when_atop_hadronic: int

    # ttH: photon record + reconstructed Higgs from di-photon.
    has_photons: bool


TT = ProcessConfig(
    name="tt",
    decay_test_index=-2,
    b_index_when_top_hadronic=1,
    b_index_when_atop_hadronic=6,
    has_photons=False,
)

TTH = ProcessConfig(
    name="tth",
    decay_test_index=-5,
    b_index_when_top_hadronic=1,
    b_index_when_atop_hadronic=6,
    has_photons=True,
)

# ttbar via SMEFTsim with complex ctG. Same final state as SM ttbar, so the
# reconstruction config is identical; only generation differs.
TT_CTG = ProcessConfig(
    name="tt_ctg",
    decay_test_index=-2,
    b_index_when_top_hadronic=1,
    b_index_when_atop_hadronic=6,
    has_photons=False,
)


# Same generation as tt_ctg but with the squared-order constraint NP^2<=1, so
# |M|^2 keeps SM + ctG interference and drops the ctG^2 term. Identical final
# state, so the reconstruction config is identical too.
TT_CTG_LIN = replace(TT_CTG, name="tt_ctg_lin")


PROCESS_CONFIGS: dict[str, ProcessConfig] = {
    "tt": TT,
    "tth": TTH,
    "tt_ctg": TT_CTG,
    "tt_ctg_lin": TT_CTG_LIN,
}


def _filesafe(s: str) -> str:
    """Make a float string filesystem-safe: '.' -> 'p' (preserves leading '-')."""
    return s.replace(".", "p")


def make_tag(process: str, label: str, semilep: int | str, nrun: int | str) -> str:
    """Canonical filename tag for one (process, label, semilep, nrun).

    Mirrors `tag_for` in data/slurm/_env.sh — must stay in sync.
    """
    if process == "tt":
        return f"top_pol_{label}_semilep_{semilep}_nrun_{nrun}"
    if process == "tth":
        return f"ttH_alpha_{label}_semilep_{semilep}_nrun_{nrun}"
    if process in ("tt_ctg", "tt_ctg_lin"):
        if "," not in label:
            raise ValueError(f"{process} label must be 're,im' (e.g. '0.5,0.3'); got {label!r}")
        re, im = label.split(",", 1)
        return f"{process}_re_{_filesafe(re)}_im_{_filesafe(im)}_semilep_{semilep}_nrun_{nrun}"
    raise ValueError(f"Unknown process: {process}")
