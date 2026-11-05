"""Driver that reads Delphes+LHE files and produces parquet-ready records.

The class is process-agnostic: per-process behavior comes from ProcessConfig.
"""

from __future__ import annotations

import glob
import os
import re

import pylhe
from tqdm import tqdm

from .config import ProcessConfig
from .cuts import (
    ETA_FATJET_MAX,
    MGG_MAX,
    MGG_MIN,
    PHOTON_ETA_MAX,
    PHOTON_PT_MIN,
    PHOTON_PT_OVER_MGG_LEAD,
    PHOTON_PT_OVER_MGG_SUB,
    PT_FATJET_MIN,
    FailCounter,
)
from .declustering import decluster
from .kinematics import delphes_to_fj, deltaR, is_in_jet_radius
from .matching import match_delphes_b_jet, match_parton_jets
from .reco import reco_hadronic_top, reco_leptonic_top, reco_w
from .writer import EventWriter


class DelphesReader:
    """Pipeline driver: ROOT + LHE -> awkward array of reconstructed events."""

    def __init__(
        self,
        process: ProcessConfig,
        *,
        clustering: str = "pnet",
        top_mass: tuple[float, float] = (165.0, 190.0),
    ):
        if clustering not in ("pnet", "ca"):
            raise ValueError(f"clustering must be 'pnet' or 'ca', got {clustering!r}")
        self.process = process
        self.clustering = clustering
        self.top_mass = top_mass
        self.writer = EventWriter(has_photons=process.has_photons)
        self.fails = FailCounter()
        # Holds fastjet ClusterSequence refs for the *current* event so they
        # outlive the writer call. Reset at the start of each event.
        self._event_clusters: list = []
        print(
            f"DelphesReader[{process.name}] initialized with "
            f"{'ParticleNet' if clustering == 'pnet' else 'CA'} declustering."
        )

    # ------------------------------------------------------------------ API

    @property
    def particle_data(self):
        """Back-compat alias for the underlying ArrayBuilder."""
        return self.writer.builder

    def read_delphes(self, filepath: str, train_data: bool = True, detect_missing: bool = False):
        files_delphes, files_parton = self._load_files(filepath)
        if detect_missing:
            self._detect_missing(files_delphes)
        print(f"Found {len(files_delphes)} root files to process.")
        for fd, fp in tqdm(
            zip(files_delphes, files_parton, strict=True),
            total=len(files_delphes),
            desc="Processing Delphes files",
        ):
            self._get_single_delphes(fd, fp, train_data)
        return self.writer.snapshot()

    def process_pair(self, delphes_file: str, lhe_file: str, train_data: bool):
        """Process a single (delphes, lhe) pair. Used by the CLI."""
        self._get_single_delphes(delphes_file, lhe_file, train_data)

    def change_clustering(self):
        self.clustering = "ca" if self.clustering == "pnet" else "pnet"
        print(
            f"Changed clustering strategy. Now using "
            f"{'ParticleNet' if self.clustering == 'pnet' else 'CA'} declustering."
        )

    def inspect_fails(self):
        self.fails.report()

    # --------------------------------------------------------- file helpers

    @staticmethod
    def _load_files(filepath: str):
        f_delphes = glob.glob(os.path.join(filepath, "**", "*.root"), recursive=True)
        f_parton = glob.glob(os.path.join(filepath, "**", "*.lhe"), recursive=True)
        delphes_sorted = sorted(
            [f.split("/")[-1].split("_")[1] for f in f_delphes],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        files_delphes = [f"{filepath}/tag_{n}_delphes_events.root" for n in delphes_sorted]
        parton_sorted = sorted(
            [f.split("/")[-1].split("_")[2].split(".")[0] for f in f_parton],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        files_parton = [f"{filepath}/unweighted_events_{n}.lhe" for n in parton_sorted]
        if parton_sorted != delphes_sorted:
            raise ValueError("Mismatch between Delphes and Parton files.")
        return files_delphes, files_parton

    @staticmethod
    def _detect_missing(files):
        files_sorted = sorted(
            (f.split("/")[-1].split("_")[1] for f in files),
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        expected = set(range(1, int(files_sorted[-1]) + 1))
        missing = expected - {int(f) for f in files_sorted}
        print(f"Expected {files_sorted[-1]} files. Missing files: {sorted(missing)}")

    # -------------------------------------------------------------- per-file

    def _get_single_delphes(self, file, file_parton, train_data):
        import os  # noqa: PLC0415

        # IMPORTANT: import & pre-warm fastjet BEFORE loading libDelphes.so.
        # Delphes statically links its own copy of fastjet, and gSystem.Load
        # below pulls those symbols in with RTLD_GLOBAL. If fastjet_cxx's
        # PLT entries are still unresolved at that point, they'll get bound
        # to Delphes's fastjet — whose PseudoJet has a different ABI —
        # and the SWIG wrapper crashes with std::length_error from
        # vector::reserve when constructing a ClusterSequence.
        import fastjet as _fj_prewarm  # noqa: PLC0415

        _pj = _fj_prewarm.PseudoJet(1.0, 0.0, 0.0, 1.0)
        _jd = _fj_prewarm.JetDefinition(_fj_prewarm.cambridge_algorithm, 0.4)
        _cs = _fj_prewarm.ClusterSequence([_pj], _jd)
        _ = _cs.inclusive_jets()
        del _cs, _jd, _pj, _fj_prewarm

        import ROOT  # noqa: PLC0415  — ROOT import is heavy; defer until needed.

        # Without an explicit Load + class-dictionary touch, PyROOT returns
        # TClonesArray entries as bare TObject and `j.BTag` fails. The path
        # comes from data/slurm/_env.sh; falls back to the legacy relative
        # path that worked when CWD was data/.
        delphes_lib = os.environ.get("DELPHES_LIB", "MG/Delphes/libDelphes.so")
        # ROOT 6.36's stricter autoloader needs to find Delphes/ExRootAnalysis
        # headers to deserialize the TClonesArray classes. Point cling at them.
        delphes_dir = os.path.dirname(os.path.abspath(delphes_lib))
        ROOT.gInterpreter.AddIncludePath(delphes_dir)
        ROOT.gInterpreter.AddIncludePath(os.path.join(delphes_dir, "external"))
        ROOT.gSystem.Load(delphes_lib)
        _ = ROOT.Jet  # force class-dict load; see note above

        try:
            in_file = ROOT.TFile.Open(file, "READ")
            delphes_tree = in_file.Get("Delphes")
            # Materialize the pylhe generator up-front. Iterating it in
            # lockstep with delphes_tree.GetEntry has been observed to
            # interact badly with PyROOT 6.36's TClonesArray refill.
            parton_info = list(pylhe.read_lhe_with_attributes(file_parton))
            self.fails.n_events += delphes_tree.GetEntries()
            print(f"Found {delphes_tree.GetEntries()} entries in file {file}")
        except Exception:
            print(f"No event found in file {file}.")
            return

        # Cache references to the tree-owned TClonesArrays once; the tree
        # refills them in place on each GetEntry. Re-fetching via attribute
        # access each entry creates fresh PyROOT wrappers in 6.36 and crashes
        # with vector::reserve on the second deserialize.
        n_entries = delphes_tree.GetEntries()
        delphes_tree.GetEntry(0)
        fat_jets = delphes_tree.FatJet
        jets = delphes_tree.Jet
        electrons = delphes_tree.Electron
        muons = delphes_tree.Muon
        met_arr = delphes_tree.MissingET
        photons = delphes_tree.Photon

        for i in tqdm(range(min(n_entries, len(parton_info)))):
            delphes_tree.GetEntry(i)
            self._process_event(
                fat_jets,
                jets,
                electrons,
                muons,
                met_arr,
                photons,
                parton_info[i],
                train_data,
            )

    def _process_event(
        self,
        rec_fat_jets,
        rec_jets,
        iso_electrons,
        iso_muons,
        met,
        photons,
        event_parton,
        train_data,
    ):
        # Release any fastjet clusters from the previous event. This is the
        # earliest safe point: the writer has already flushed the previous
        # event's subjet constituents by now.
        self._event_clusters.clear()
        if rec_fat_jets.GetEntriesFast() == 0:
            self.fails.bump("missing fatjet")
            return
        if iso_electrons.GetEntriesFast() + iso_muons.GetEntriesFast() != 1:
            self.fails.bump("not having exactly one isolated lepton")
            return
        n_jets = rec_jets.GetEntriesFast()
        if n_jets == 0 or sum(rec_jets.At(j).BTag != 0 for j in range(n_jets)) < 2:
            self.fails.bump("missing b-tagged jet")
            return

        rec_leading, rec_subleading = self._order_fat_jets(rec_fat_jets)
        if rec_leading.PT < PT_FATJET_MIN or abs(rec_leading.Eta) > ETA_FATJET_MAX:
            self.fails.bump("kinematic cuts")
            return

        photons_out = None
        if self.process.has_photons:
            photons_out = self._photon_cuts(photons)
            if photons_out is None:
                return

        partons = event_parton.particles[2:]
        b_tagged_jets = [rec_jets.At(j) for j in range(n_jets) if rec_jets.At(j).BTag == 1]
        if iso_electrons.GetEntriesFast() == 1:
            iso_lepton = iso_electrons.At(0)
        else:
            iso_lepton = iso_muons.At(0)
        missing_et = met.At(0)

        if rec_subleading is not None:
            in_lead = is_in_jet_radius(iso_lepton, rec_leading)
            in_sub = is_in_jet_radius(iso_lepton, rec_subleading)
            if in_lead + in_sub != 1:
                self.fails.bump("unclear lepton affiliation")
                return
            hadronic_top_cand = rec_subleading if in_lead else rec_leading
        else:
            hadronic_top_cand = rec_leading

        hadronic_top, hadronic_top_subjets = self._hadronic_top_reco(
            hadronic_top_cand,
            partons,
            b_tagged_jets,
            train_data,
        )
        if hadronic_top_subjets is None:
            return

        if deltaR(b_tagged_jets[0], hadronic_top_subjets[0]) < deltaR(
            b_tagged_jets[1], hadronic_top_subjets[0]
        ):
            b_lep_cand = b_tagged_jets[1]
        else:
            b_lep_cand = b_tagged_jets[0]

        leptonic_top, lepton = self._leptonic_top_reco(
            iso_lepton,
            missing_et,
            b_lep_cand,
        )
        if lepton is None:
            return

        self.writer.add_event(
            tophad=hadronic_top,
            subjets=hadronic_top_subjets,
            toplep=leptonic_top,
            lepton=lepton,
            lep_charge=iso_lepton.Charge,
            photons=photons_out,
            train_data=train_data,
        )

    # ------------------------------------------------- sub-steps

    @staticmethod
    def _order_fat_jets(rec_fat_jets):
        if rec_fat_jets.GetEntriesFast() < 2:
            return rec_fat_jets.At(0), None
        j0, j1 = rec_fat_jets.At(0), rec_fat_jets.At(1)
        return (j0, j1) if j0.PT > j1.PT else (j1, j0)

    def _photon_cuts(self, photons):
        if photons.GetEntriesFast() < 2:
            self.fails.bump("missing photons")
            return None
        lead, sub = photons.At(0), photons.At(1)
        assert lead.PT >= sub.PT, "Photons are not ordered by pT in Delphes output."
        if (
            lead.PT < PHOTON_PT_MIN
            or abs(lead.Eta) > PHOTON_ETA_MAX
            or sub.PT < PHOTON_PT_MIN
            or abs(sub.Eta) > PHOTON_ETA_MAX
        ):
            self.fails.bump("photon pT/eta cuts")
            return None
        m_gaga = (lead.P4() + sub.P4()).M()
        if lead.PT / m_gaga < PHOTON_PT_OVER_MGG_LEAD or sub.PT / m_gaga < PHOTON_PT_OVER_MGG_SUB:
            self.fails.bump("photon ISR cuts")
            return None
        if m_gaga < MGG_MIN or m_gaga > MGG_MAX:
            self.fails.bump("m_gamma_gamma cuts")
            return None
        return [lead, sub]

    def _hadronic_top_reco(self, top_cand, partons, b_tagged_jets, train_data):
        cluster, subjets = decluster(top_cand.Constituents, strategy=self.clustering)
        # Keep the fastjet ClusterSequence alive until the writer has flushed
        # this event — subjets reference it for .constituents().
        self._event_clusters.append(cluster)
        if subjets is None:
            self.fails.bump("declustering")
            return None, None

        top_subjets = reco_hadronic_top(subjets, self.top_mass)
        if top_subjets is None:
            self.fails.bump("hadronic top mass reconstruction")
            return None, None

        if train_data:
            tagged = match_parton_jets(partons, top_subjets, self.process, max_dr=0.4)
        else:
            tagged = match_delphes_b_jet(b_tagged_jets, top_subjets)
        if tagged is None:
            self.fails.bump("parton jet matching")
            return None, None

        if reco_hadronic_top(tagged, self.top_mass) is None:
            self.fails.bump("top mass after matching")
            return None, None

        top = tagged[0] + tagged[1] + tagged[2]
        return top, tagged

    def _leptonic_top_reco(self, lepton_cand, neutrino_cand, b_cand):
        lepton = delphes_to_fj(lepton_cand)
        w = reco_w(lepton, neutrino_cand)
        top = reco_leptonic_top(w, lepton, b_cand, self.top_mass)
        if top is None:
            self.fails.bump("leptonic top mass reconstruction")
            return None, None
        return top, lepton
