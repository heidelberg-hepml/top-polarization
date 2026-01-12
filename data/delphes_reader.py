import glob
import os
import math
import re
from tqdm import tqdm
import awkward as ak
import itertools
import pylhe
from pylorentz import Momentum4
import ROOT
import fastjet as fj
ROOT.gSystem.Load('/bigwork/nhbbmene/PostDoc/MadGraph/MG5_aMC_v3_6_4/Delphes/libDelphes.so')

class DelphesReader():

    def __init__(self, pnet=True, top_mass=[165,190]):

        self.jet_data = ak.ArrayBuilder()
        self.pnet_clustering = pnet
        self.top_mass = top_mass
        self.NEvents = 0
        self.NFatjetFail = 0
        self.NBTagFail = 0
        self.KinematicsFail = 0
        self.DeclusteringFail = 0
        self.TopmassFail = 0
        self.JetMatchFail = 0
        self.FailFail = 0

        print(f"DelphesReader initialized with {'ParticleNet' if self.pnet_clustering else 'CA'} declustering strategy.")

    def _load_files(self, filepath):

        f_delphes = glob.glob(os.path.join(filepath, '**', '*.root'), recursive=True)
        f_parton = glob.glob(os.path.join(filepath, '**', '*.lhe'), recursive=True)

        delphes_sorted = sorted([f.split('/')[-1].split('_')[1] for f in f_delphes], key=lambda x: int(re.search(r'\d+', x).group()))
        files_delphes = [f"{filepath}/tag_{n}_delphes_events.root" for n in delphes_sorted]

        parton_sorted = sorted([f.split('/')[-1].split('_')[2].split('.')[0] for f in f_parton], key=lambda x: int(re.search(r'\d+', x).group()))
        files_parton = [f"{filepath}/unweighted_events_{n}.lhe" for n in parton_sorted]

        if parton_sorted != delphes_sorted:
            raise ValueError("Mismatch between Delphes and Parton files.")

        return files_delphes, files_parton

    def _detect_missing(self, files):
        files_sorted = [f.split('/')[-1].split('_')[1] for f in files]
        files_sorted = sorted(files_sorted, key=lambda x: int(re.search(r'\d+', x).group()))
        expected = set(i for i in range(1, int(files_sorted[-1]) + 1))
        missing = expected - set(int(f) for f in files_sorted)
        print(f"Expected {files_sorted[-1]} files. Missing files: {sorted(list(missing))}")

    def _deltaR(self, p1, p2, train_data):

        '''
        Efficient deltaR calculation between two jets.
        '''

        if train_data:
            dphi = abs(p1.phi() - p2.phi)
            deta = p1.eta() - p2.eta
        else:
            dphi = abs(p1.phi() - p2.Phi)
            deta = p1.eta() - p2.Eta

        if dphi > math.pi:
            dphi = dphi - 2 * math.pi

        return math.sqrt(deta**2 + dphi**2)

    def _inv_mass(self, subjets):

        '''
        Calculate the invariant mass of multiple subjets.
        '''

        s = subjets[0]
        for j in subjets[1:]:
            s = s + j
        return s.m()

    def _get_fj_parents(self, cluster, jet):

        '''
        Get the two parent jets of a given jet in a FastJet ClusterSequence.
        '''

        p1 = fj.PseudoJet()
        p2 = fj.PseudoJet()
        check_for_parents = cluster.has_parents(jet, p1, p2)
        if not check_for_parents:
            return None, None
        return p1, p2

    def _decluster(self, constits, m_max=30, threshhold=0.05, R=1.5):

        '''
        Declustering of FatJet into subjets.

        The strategy used for declustering is decided by pnet_clustering.
            True:  Use the strategy from "Hadronic Top Quark Polarimetry with ParticleNet" (https://arxiv.org/pdf/2407.01663)
            False: Use the strategy from the Cambride-Aachen (CA) algorithm (https://cds.cern.ch/record/1194489/files/JME-09-001-pas.pdf)

        m_max: Maximum mass for subjets in the ParticleNet declustering strategy. Default is 30 GeV.

        threshhold: Fractional pT threshhold for subjets in the CA declustering strategy. Default is 0.05.

        R: Jet radius parameter for the initial clustering. Has to match the jet radius in the Delphes file. Default is 1.5.
        '''

        jet_def = fj.JetDefinition(fj.cambridge_algorithm, R)
        fj_particles = []
        for constit in constits:
            charge = constit.Charge if hasattr(constit, 'Charge') else 0
            pid = constit.PID if hasattr(constit, 'PID') else 0
            p4 = constit.P4()
            fj_part = fj.PseudoJet(p4.Px(), p4.Py(), p4.Pz(), p4.E())
            fj_part.set_user_index(int(pid*3 + (charge+1)))
            fj_particles.append(fj_part)

        cluster = fj.ClusterSequence(fj_particles, jet_def)
        jet_fj = max(cluster.inclusive_jets(), key=lambda j: j.pt())

        if self.pnet_clustering:

            def rec_pnet(child):

                '''
                Recursive declustering according to https://arxiv.org/pdf/2407.01663
                '''

                parent1, parent2 = self._get_fj_parents(cluster, child)
                if parent1 is None or parent2 is None:
                    return [child]

                if parent2.m() > parent1.m():
                    parent1, parent2 = parent2, parent1

                if parent1.m() < m_max:
                    return [parent1, parent2]

                if parent1.m() > m_max and parent2.m() < m_max:
                    return rec_pnet(parent1) + [parent2]

                if parent2.m() > m_max:
                    return rec_pnet(parent1) + rec_pnet(parent2)

                return [child]

            subjets = rec_pnet(jet_fj)

        else:

            def rec_CA(child, pt_hard, depth=0):

                '''
                Recursive declustering according to https://cds.cern.ch/record/1194489/files/JME-09-001-pas.pdf
                '''

                parent1, parent2 = self._get_fj_parents(cluster, child)
                if parent1 is None or parent2 is None:
                    return [child]

                if parent2.pt() > parent1.pt():
                    parent1, parent2 = parent2, parent1

                if parent1.pt() < threshhold * pt_hard:
                    return [child]

                if parent1.pt() > threshhold * pt_hard and parent2.pt() < threshhold * pt_hard:
                    return rec_CA(parent1, pt_hard=pt_hard, depth=depth)

                if parent2.pt() > threshhold * pt_hard:
                    if depth == 0:
                        return rec_CA(parent1, pt_hard=pt_hard, depth=1) + rec_CA(parent2, pt_hard=pt_hard, depth=1)
                    if depth == 1:
                        return [parent1, parent2]

                return [child]

            subjets = rec_CA(jet_fj, pt_hard=jet_fj.pt())

        if len(subjets) <= 2:
            return None

        subjets = sorted(subjets, key=lambda s: s.pt(), reverse=True)

        return subjets[:4]

    def _top_mass_reco(self, subjets, top_mass = 173):

        '''
        Check if the subjets can reconstruct a top quark mass within the allowed mass window.

        A mass window of [165, 190] (default) corresponds to https://arxiv.org/pdf/2407.01663

        A mass window of [120, 220] corresponds to https://arxiv.org/pdf/1908.07305
        '''

        best_subjets = subjets
        m = self._inv_mass(subjets)
        best_diff = abs(m - top_mass)
        best_mass = m

        for candidates in itertools.combinations(subjets, 3):
            m = self._inv_mass(candidates)
            diff = abs(m - top_mass)
            if diff < best_diff:
                best_diff = diff
                best_mass = m
                best_subjets = candidates

        if best_mass < int(self.top_mass[0]) or best_mass > int(self.top_mass[1]):
            return None

        return best_subjets

    def _match_parton_jets(self, partons, subjets, max_dr=0.4):

        '''
        Match the subjets to parton-level objects for the training data.
        '''

        matches = []

        ids = [p.id for p in partons]
        if ids[-2] in [11, -11, 13, -13]:
            hadronic = "top"
        elif ids[-2] in [1, -1, 2, -2, 3, -3, 4, -4]:
            hadronic = "atop"
        else:
            raise ValueError(f"Parton-level objets do not match to expected IDs. The ID of the lepton/quark is {ids[-2]}.")

        if hadronic == "top":
            b_hadronic = partons[1]
        else:
            b_hadronic = partons[6]
        u_index, d_index = next((i for i, p in enumerate(partons) if abs(p.id) == 2), None), next((i for i, p in enumerate(partons) if abs(p.id) == 1), None)
        if u_index is None or d_index is None:
            raise ValueError("Could not find both up-type and down-type quarks in parton-level objects.")
        u_hadronic, d_hadronic = partons[u_index], partons[d_index]

        for tag, ps, uidx in zip(["b", "u", "d"], [b_hadronic, u_hadronic, d_hadronic], [5, 2, 1]):
            drs = []
            for sj in subjets:
                drs.append(self._deltaR(sj, Momentum4(ps.e, ps.px, ps.py, ps.pz), train_data=True))
            min_dr = min(drs)
            min_idx = drs.index(min_dr)
            subjets[min_idx].set_user_index(uidx)
            matches.append((tag, min_idx, min_dr))

        if len([match[1] for match in matches]) != len(set([match[1] for match in matches])):
            return None
        if any(match[2] > max_dr for match in matches):
            return None

        b_cand = subjets[matches[0][1]]
        subjets = [sj for c, sj in enumerate(subjets) if c in [m[1] for m in matches]]

        return [b_cand] + [sj for sj in subjets if sj != b_cand]

    def _match_delphes_b_jet(self, b_tagged_jets, subjets, max_dr=0.1):

        '''
        Check if one of the subjets is close to a b-tagged jet and if yes, give it a b-tag.
        '''

        min_dr = 1e6
        sj_cand = None

        for sj in subjets:
            for b in b_tagged_jets:
                dr = self._deltaR(sj, b, train_data=False)
                if dr < min_dr:
                    min_dr = dr
                    sj_cand = sj

        if min_dr <= max_dr:
            return [sj_cand] + [sj for sj in subjets if sj != sj_cand]

        return None

    def _build_output(self, fatjet, subjets, train_data):

        '''
        Adds a record for a single event to the output awkward array.
        '''

        self.jet_data.begin_record()
        ## Event record ##

        self.jet_data.field("FatJet")
        self.jet_data.begin_record()
        ## FatJet record ##

        self.jet_data.field("pt").real(fatjet.PT)
        self.jet_data.field("phi").real(fatjet.Phi)
        self.jet_data.field("eta").real(fatjet.Eta)
        self.jet_data.field("mass").real(fatjet.Mass)

        ## FatJet record ##
        self.jet_data.end_record()

        self.jet_data.field("Subjets")
        self.jet_data.begin_list()
        for c, subjet in enumerate(subjets):
            self.jet_data.begin_record()
            ## Subjet records ##

            self.jet_data.field("pt").real(subjet.pt())
            self.jet_data.field("phi").real(subjet.phi())
            self.jet_data.field("eta").real(subjet.eta())
            self.jet_data.field("mass").real(subjet.m())

            if train_data:
                self.jet_data.field("tag").real(subjet.user_index())
            else:
                if c == 0:
                    self.jet_data.field("tag").real(5)
                else:
                    self.jet_data.field("tag").real(0)

            self.jet_data.field("Constituents")
            self.jet_data.begin_list()
            for constit in subjet.constituents():
                self.jet_data.begin_record()
                ## Subjet constits records ##

                self.jet_data.field("pt").real(constit.pt())
                self.jet_data.field("phi").real(constit.phi())
                self.jet_data.field("eta").real(constit.eta())
                self.jet_data.field("mass").real(constit.m())
                self.jet_data.field("charge").real((constit.user_index() % 3) - 1)
                self.jet_data.field("pid").real(constit.user_index() // 3)

                ## Subjet constits records ##
                self.jet_data.end_record()
            self.jet_data.end_list()

            ## Subjet records ##
            self.jet_data.end_record()
        self.jet_data.end_list()

        ## Event record ##
        self.jet_data.end_record()

    def _get_single_delphes(self, file, file_parton, train_data):

        try:
            inFile = ROOT.TFile.Open(file, 'READ')
            delphesTree = inFile.Get('Delphes')
            partonInfo = pylhe.read_lhe_with_attributes(file_parton)
            self.NEvents += delphesTree.GetEntries()
            # print(f"{delphesTree.GetEntries()} events found in file {file.split('temp_')[-1].split('/')[0]}.")
        except:
            print(f"No event found in file {file}.")
            return

        rec_fat_jets = ROOT.TClonesArray("Jet")
        rec_jets = ROOT.TClonesArray("Jet")
        delphesTree.SetBranchAddress("FatJet", rec_fat_jets)
        delphesTree.SetBranchAddress("Jet", rec_jets)

        for event, event_parton in zip(delphesTree, partonInfo):

            # Cut: Require at least 1 FatJet
            if rec_fat_jets.GetEntriesFast() == 0:
                self.NFatjetFail += 1
                continue

            # Cut: Require at least 1 B-tagged Jet
            if rec_jets.GetEntriesFast() == 0 or all(j.BTag == 0 for j in rec_jets):
                self.NBTagFail += 1
                continue

            if rec_fat_jets.GetEntriesFast() < 2:
                rec_leading_jet = rec_fat_jets.At(0)
            else:
                if rec_fat_jets.At(0).PT > rec_fat_jets.At(1).PT:
                    rec_leading_jet = rec_fat_jets.At(0)
                else:
                    rec_leading_jet = rec_fat_jets.At(1)

            # Cut: Require minimum pT and maximum eta for leading fat jet
            if rec_leading_jet.PT < 250 or abs(rec_leading_jet.Eta) > 3.:
                self.KinematicsFail += 1
                continue

            # Decluster hadronic fat jet into subjets
            subjets = self._decluster(rec_leading_jet.Constituents)
            if subjets is None:
                self.DeclusteringFail += 1
                continue

            # Reconstruct top mass from subjets
            top_subjets = self._top_mass_reco(subjets)
            if top_subjets is None:
                self.TopmassFail += 1
                continue

            # Match hadronic subjets to parton jets (training data) or apply b-tagging (test data)
            if train_data:
                partons = event_parton.particles[2:]
                top_subjets_tagged = self._match_parton_jets(partons, top_subjets)
            else:
                b_tagged_jets = [j for j in rec_jets if j.BTag == 1]
                top_subjets_tagged = self._match_delphes_b_jet(b_tagged_jets, top_subjets)

            if top_subjets_tagged is None:
                self.JetMatchFail += 1
                continue

            jets_failfail = self._top_mass_reco(top_subjets_tagged)
            if jets_failfail is None:
                self.FailFail += 1
                continue

            self._build_output(rec_leading_jet, top_subjets_tagged, train_data)

    def change_clustering(self):
        self.pnet_clustering = not self.pnet_clustering
        print(f"Changed clustering strategy. Now using {'ParticleNet' if self.pnet_clustering else 'CA'} declustering.")

    def inspect_fails(self):

        print(f"Fails due to missing fatjet: {self.NFatjetFail}")
        print(f"Fails due to missing b-tagged jet: {self.NBTagFail}")
        print(f"Fails due to kinematic cuts: {self.KinematicsFail}")
        print(f"Fails due to declustering: {self.DeclusteringFail}")
        print(f"Fails due to top mass reconstruction: {self.TopmassFail}")
        print(f"Fails due to parton jet matching: {self.JetMatchFail}")
        print(f"Fails due to top mass after matching: {self.FailFail}")
        print(f"Total events processed: {self.NEvents}")
        print(f"Surviving number of events: {len(self.jet_data.snapshot().FatJet.pt)}")

    def read_delphes(self, filepath, train_data=True, detect_missing=False):

        files_delphes, files_parton = self._load_files(filepath)

        if detect_missing:
            self._detect_missing(files_delphes)

        print(f"Found {len(files_delphes)} root files to process.")

        for f_d, f_p in tqdm(zip(files_delphes, files_parton), total=len(files_delphes), desc="Processing Delphes files"):
            self._get_single_delphes(f_d, f_p, train_data)

        return self.jet_data.snapshot()
