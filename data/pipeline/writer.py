"""Write reconstructed events into an awkward ArrayBuilder."""

from __future__ import annotations

import awkward as ak


class EventWriter:
    """Thin wrapper around `ak.ArrayBuilder` that knows the per-process schema."""

    def __init__(self, has_photons: bool):
        self.has_photons = has_photons
        self.builder = ak.ArrayBuilder()

    def snapshot(self):
        return self.builder.snapshot()

    def add_event(self, *, tophad, subjets, toplep, lepton, lep_charge, photons, train_data: bool):
        b = self.builder
        b.begin_record()

        self._write_top(b, "TopLep", toplep)
        self._write_lepton(b, lepton, lep_charge)
        self._write_top(b, "TopHad", tophad)
        self._write_subjets(b, subjets, train_data)
        if self.has_photons:
            self._write_photons(b, photons)

        b.end_record()

    @staticmethod
    def _write_top(b, name, top):
        b.field(name)
        b.begin_record()
        b.field("pt").real(top.pt())
        b.field("phi").real(top.phi())
        b.field("eta").real(top.eta())
        b.field("mass").real(top.m())
        b.end_record()

    @staticmethod
    def _write_lepton(b, lepton, charge):
        b.field("Lepton")
        b.begin_record()
        b.field("pt").real(lepton.pt())
        b.field("phi").real(lepton.phi())
        b.field("eta").real(lepton.eta())
        b.field("charge").real(charge)
        b.end_record()

    @staticmethod
    def _write_subjets(b, subjets, train_data: bool):
        b.field("Subjets")
        b.begin_list()
        for c, subjet in enumerate(subjets):
            b.begin_record()
            b.field("pt").real(subjet.pt())
            b.field("phi").real(subjet.phi())
            b.field("eta").real(subjet.eta())
            b.field("mass").real(subjet.m())
            if train_data:
                b.field("tag").real(subjet.user_index())
            else:
                b.field("tag").real(5 if c == 0 else 0)

            b.field("Constituents")
            b.begin_list()
            for constit in subjet.constituents():
                b.begin_record()
                b.field("pt").real(constit.pt())
                b.field("phi").real(constit.phi())
                b.field("eta").real(constit.eta())
                b.field("mass").real(constit.m())
                b.field("charge").real((constit.user_index() % 3) - 1)
                b.field("pid").real(constit.user_index() // 3)
                b.end_record()
            b.end_list()
            b.end_record()
        b.end_list()

    @staticmethod
    def _write_photons(b, photons):
        for name, ph in (("photon_leading", photons[0]), ("photon_subleading", photons[1])):
            b.field(name)
            b.begin_record()
            b.field("pt").real(ph.PT)
            b.field("phi").real(ph.Phi)
            b.field("eta").real(ph.Eta)
            b.field("mass").real(ph.P4().M())
            b.end_record()

        b.field("higgs")
        b.begin_record()
        higgs = photons[0].P4() + photons[1].P4()
        b.field("pt").real(higgs.Pt())
        b.field("phi").real(higgs.Phi())
        b.field("eta").real(higgs.Eta())
        b.field("mass").real(higgs.M())
        b.end_record()
