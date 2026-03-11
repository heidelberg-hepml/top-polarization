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
ROOT.gSystem.Load('MG/Delphes/libDelphes.so')
print(ROOT.Jet) # necessary to avoid Delphes not finding header files

class DelphesReader():

    def __init__(self, pnet=True, top_mass=[165,190], ttH=False):

        self.jet_data = ak.ArrayBuilder()
        self.pnet_clustering = pnet
        self.top_mass = top_mass
        self.ttH = ttH
        self.NEvents = 0
        self.NFatjetFail = 0
        self.NLepFail = 0
        self.NBTagFail = 0
        self.KinematicsFail = 0
        self.LeptonIdentificationFail = 0
        self.BIdentificationFail = 0
        self.DeclusteringFail = 0
        self.TopMassFailHad = 0
        self.TopMassFailLep = 0
        self.JetMatchFail = 0
        self.FailFail = 0
        self.NPhotonFail = 0
        self.NPhotonPtEtaFail = 0
        self.NPhotonISRFail = 0
        self.NPhotonMgamgamFail = 0

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

    def _delphes_to_fj(self, obj):
        px = obj.PT * math.cos(obj.Phi)
        py = obj.PT * math.sin(obj.Phi)
        pz = obj.PT * math.sinh(obj.Eta)
        E  = obj.PT * math.cosh(obj.Eta)
        return fj.PseudoJet(px, py, pz, E)

    def _phi(self, p):
        return (
            p.phi() if callable(getattr(p, "phi", None)) else
            p.phi   if hasattr(p, "phi") else
            p.Phi
        )

    def _eta(self, p):
        return (
            p.eta() if callable(getattr(p, "eta", None)) else
            p.eta   if hasattr(p, "eta") else
            p.Eta
        )

    def _deltaR(self, p1, p2):

        '''
        Efficient deltaR calculation between two jets.
        '''

        dphi = abs(self._phi(p1) - self._phi(p2))
        deta = self._eta(p1) - self._eta(p2)

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

    def _is_in_jet_radius(self, particle, jet):

        dist = self._deltaR(particle, jet)
        
        if dist > 1.5:
            return False

        return True

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

    def _top_mass_reco_had(self, subjets, mtop = 173):

        '''
        Check if the subjets can reconstruct a top quark mass within the allowed mass window.

        A mass window of [165, 190] (default) corresponds to https://arxiv.org/pdf/2407.01663

        A mass window of [120, 220] corresponds to https://arxiv.org/pdf/1908.07305
        '''

        best_subjets = subjets
        m = self._inv_mass(subjets)
        best_diff = abs(m - mtop)
        best_mass = m

        for candidates in itertools.combinations(subjets, 3):
            m = self._inv_mass(candidates)
            diff = abs(m - mtop)
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
        test_id = -5 if self.ttH else -2
        if ids[test_id] in [11, -11, 13, -13]: 
            hadronic = "top"
        elif ids[test_id] in [1, -1, 2, -2, 3, -3, 4, -4]:
            hadronic = "atop"
        else:
            raise ValueError(f"Parton-level objets do not match to expected IDs. The ID of the lepton/quark is {ids[test_id]}.")

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
                drs.append(self._deltaR(sj, Momentum4(ps.e, ps.px, ps.py, ps.pz)))
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
                dr = self._deltaR(sj, b)
                if dr < min_dr:
                    min_dr = dr
                    sj_cand = sj

        if min_dr <= max_dr:
            return [sj_cand] + [sj for sj in subjets if sj != sj_cand]

        return None

    def _reconstruct_w(self, l, neutrino, w_mass=80):
        pxnu, pynu = neutrino.MET * math.cos(neutrino.Phi), neutrino.MET * math.sin(neutrino.Phi)

        coeff = (w_mass**2/2 + l.px()*pxnu + l.py()*pynu)
        transverse_cond = coeff**2 - (l.pt())**2 * (neutrino.MET)**2

        if transverse_cond < 0:
            met_corrected = coeff / l.pt()
            pxnu, pynu = met_corrected * math.cos(neutrino.Phi), met_corrected * math.sin(neutrino.Phi)
            pznu = coeff * l.pz() / (l.pt())**2
            enu = math.sqrt(met_corrected**2 + pznu**2)
        else:
            pznu1 = coeff * l.pz() / (l.pt())**2 + (l.e() / (l.pt())**2) * math.sqrt( transverse_cond )
            pznu2 = coeff * l.pz() / (l.pt())**2 - (l.e() / (l.pt())**2) * math.sqrt( transverse_cond )
            pznu1, pznu2 = pznu1.real, pznu2.real
            pznu = pznu1 if abs(pznu1) <= abs(pznu2) else pznu2
            enu = math.sqrt(neutrino.MET**2 + pznu**2)

        nu = fj.PseudoJet(pxnu, pynu, pznu, enu)

        w = l + nu

        return w

    def _top_mass_reco_lep(self, w, l, b_quark):

        b = self._delphes_to_fj(b_quark)

        t1 = w + b
        if t1.m() < int(self.top_mass[0]) or t1.m() > int(self.top_mass[1]):
            t2 = w + b - l
            if t2.m() < int(self.top_mass[0]) or t2.m() > int(self.top_mass[1]):
                return None
            else:
                return t2
        return t1

    def _hadronic_top_reconstruction(self, top_cand, partons, b_tagged_jets, train_data=True):

        # Decluster hadronic fat jet into subjets
        subjets = self._decluster(top_cand.Constituents)
        if subjets is None:
            self.DeclusteringFail += 1
            return None, None

        # Reconstruct top mass from subjets
        top_subjets = self._top_mass_reco_had(subjets)
        if top_subjets is None:
            self.TopMassFailHad += 1
            return None, None

        # Match hadronic subjets to parton jets (training data) or apply b-tagging (test data)
        if train_data:
            top_subjets_tagged = self._match_parton_jets(partons, top_subjets, max_dr=0.4)
        else:
            top_subjets_tagged = self._match_delphes_b_jet(b_tagged_jets, top_subjets)

        if top_subjets_tagged is None:
            self.JetMatchFail += 1
            return None, None

        jets_failfail = self._top_mass_reco_had(top_subjets_tagged)
        if jets_failfail is None:
            self.FailFail += 1
            return None, None

        top = top_subjets_tagged[0]+top_subjets_tagged[1]+top_subjets_tagged[2]

        return top, top_subjets_tagged

    def _leptonic_top_reconstruction(self, partons, lepton_cand, neutrino_cand, b_cand):

        lepton = self._delphes_to_fj(lepton_cand)

        # Reconstruct neutrino pz with w mass criteria
        w_boson_cand = self._reconstruct_w(lepton, neutrino_cand)

        # Reconstruct top mass
        top = self._top_mass_reco_lep(w_boson_cand, lepton, b_cand)
        if top is None:
            self.TopMassFailLep += 1
            return None, None

        return top, lepton

    def _build_output(self, tophad, subjets, toplep, lepton, lep_charge, train_data):

        '''
        Adds a record for a single event to the output awkward array.
        '''
    
        self.jet_data.begin_record()
        ## Begin event record ##

        self.jet_data.field("TopLep")
        self.jet_data.begin_record()
        ## Begin leptonic top record ##

        self.jet_data.field("pt").real(toplep.pt())
        self.jet_data.field("phi").real(toplep.phi())
        self.jet_data.field("eta").real(toplep.eta())
        self.jet_data.field("mass").real(toplep.m())

        ## End leptonic top record ##
        self.jet_data.end_record()

        self.jet_data.field("Lepton")
        self.jet_data.begin_record()
        ## Begin lepton record ##

        # Add lepton charge to flip sign in polarization angle
        self.jet_data.field("pt").real(lepton.pt())
        self.jet_data.field("phi").real(lepton.phi())
        self.jet_data.field("eta").real(lepton.eta())
        self.jet_data.field("charge").real(lep_charge)

        ## End lepton record ##
        self.jet_data.end_record()

        self.jet_data.field("TopHad")
        self.jet_data.begin_record()
        ## Begin hadronic top record ##

        self.jet_data.field("pt").real(tophad.pt())
        self.jet_data.field("phi").real(tophad.phi())
        self.jet_data.field("eta").real(tophad.eta())
        self.jet_data.field("mass").real(tophad.m())

        ## End hadronic top record ##
        self.jet_data.end_record()

        self.jet_data.field("Subjets")
        self.jet_data.begin_list()
        for c, subjet in enumerate(subjets):
            self.jet_data.begin_record()
            ## Begin subjet records ## 

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
                ## Begin subjet constits records ##

                self.jet_data.field("pt").real(constit.pt())
                self.jet_data.field("phi").real(constit.phi())
                self.jet_data.field("eta").real(constit.eta())
                self.jet_data.field("mass").real(constit.m())
                self.jet_data.field("charge").real((constit.user_index() % 3) - 1)
                self.jet_data.field("pid").real(constit.user_index() // 3)

                ## End subjet constits records ##
                self.jet_data.end_record()
            self.jet_data.end_list()

            ## End subjet records ## 
            self.jet_data.end_record()
        self.jet_data.end_list()

        ## End event record ##
        self.jet_data.end_record()

    def _get_single_delphes(self, file, file_parton, train_data):

        try:
            inFile = ROOT.TFile.Open(file, 'READ')
            delphesTree = inFile.Get('Delphes')
            partonInfo = pylhe.read_lhe_with_attributes(file_parton)
            self.NEvents += delphesTree.GetEntries()
            print(f"Found {delphesTree.GetEntries()} entries in file {file}")
        except:
            print(f"No event found in file {file}.")
            return

        rec_fat_jets = ROOT.TClonesArray("Jet")
        rec_jets = ROOT.TClonesArray("Jet")
        iso_electrons = ROOT.TClonesArray("Electron")
        iso_muons = ROOT.TClonesArray("Muon")
        met = ROOT.TClonesArray("MissingET")
        photons = ROOT.TClonesArray("Photon")
        delphesTree.SetBranchAddress("FatJet", rec_fat_jets)
        delphesTree.SetBranchAddress("Jet", rec_jets)
        delphesTree.SetBranchAddress("Electron", iso_electrons)
        delphesTree.SetBranchAddress("Muon", iso_muons)
        delphesTree.SetBranchAddress("MissingET", met)
        delphesTree.SetBranchAddress("Photon", photons)

        for event, event_parton in zip(delphesTree, partonInfo):
            
            # Cut: Require at least 1 FatJet
            if rec_fat_jets.GetEntriesFast() == 0:
                self.NFatjetFail += 1
                continue

            # Cut: Require exactly one isolated lepton. Note that this is a ttbar cut. It should be relaxed to at least one lepton for ttH.
            if iso_electrons.GetEntriesFast() + iso_muons.GetEntriesFast() != 1:
                self.NLepFail += 1
                continue

            # Cut: Require at least 2 B-tagged Jets
            if rec_jets.GetEntriesFast() == 0 or sum(j.BTag != 0 for j in rec_jets) < 2:
                self.NBTagFail += 1
                continue

            if rec_fat_jets.GetEntriesFast() < 2:
                rec_leading_jet = rec_fat_jets.At(0)
                rec_subleading_jet = None
            else:
                if rec_fat_jets.At(0).PT > rec_fat_jets.At(1).PT:
                    rec_leading_jet = rec_fat_jets.At(0)
                    rec_subleading_jet = rec_fat_jets.At(1)
                else:
                    rec_leading_jet = rec_fat_jets.At(1)
                    rec_subleading_jet = rec_fat_jets.At(0)

            # Cut: Require minimum pT and maximum eta for leading fat jet 
            if rec_leading_jet.PT < 250 or abs(rec_leading_jet.Eta) > 3.:
                self.KinematicsFail += 1
                continue
            
            if self.ttH: # ttH specific cuts
                # Cut: Require two photons
                if photons.GetEntriesFast() < 2:
                    self.NPhotonFail += 1
                    continue
                leading_photon, subleading_photon = photons.At(0), photons.At(1)
                assert leading_photon.PT >= subleading_photon.PT, "Photons are not ordered by pT in Delphes output."
                
                # Cut: Require two photons to have pT > 20 GeV and |eta| < 2.5
                if leading_photon.PT < 20 or abs(leading_photon.Eta) > 2.5 or subleading_photon.PT < 20 or abs(subleading_photon.Eta) > 2.5:
                    self.NPhotonPtEtaFail += 1
                    continue
                
                # Cut: photon ISR cut, leading photon pt/mgaga > .35, and subleading photon pt/mgaga > .25
                m_gaga = (leading_photon.P4() + subleading_photon.P4()).M()
                if leading_photon.PT / m_gaga < 0.35 or subleading_photon.PT / m_gaga < 0.25:
                    self.NPhotonISRFail += 1
                    continue
                
                # Cut: 110 GeV < m_gamma_gamma < 140 GeV
                if m_gaga < 110 or m_gaga > 140:
                    self.NPhotonMgamgamFail += 1
                    continue
                 

            # Get all needed particles
            partons = event_parton.particles[2:]
            b_tagged_jets = [j for j in rec_jets if j.BTag == 1]
            if iso_electrons.GetEntriesFast() == 1:
                iso_lepton = iso_electrons.At(0)
            elif iso_muons.GetEntriesFast() == 1:
                iso_lepton = iso_muons.At(0)
            missing_et = met.At(0)

            # Choose candidate for leptonically decaying top if two FatJets are present
            if rec_subleading_jet:
                if self._is_in_jet_radius(iso_lepton, rec_leading_jet) + self._is_in_jet_radius(iso_lepton, rec_subleading_jet) != 1:
                    self.LeptonIdentificationFail += 1
                    continue

                if self._is_in_jet_radius(iso_lepton, rec_leading_jet):
                    hadronic_top_cand = rec_subleading_jet
                elif self._is_in_jet_radius(iso_lepton, rec_subleading_jet):
                    hadronic_top_cand = rec_leading_jet
            else:
                hadronic_top_cand = rec_leading_jet

            hadronic_top, hadronic_top_subjets = self._hadronic_top_reconstruction(hadronic_top_cand, partons, b_tagged_jets, train_data)
            if not hadronic_top_subjets:
                continue

            if self._deltaR(b_tagged_jets[0], hadronic_top_subjets[0]) < self._deltaR(b_tagged_jets[1], hadronic_top_subjets[0]):
                b_lep_cand = b_tagged_jets[1]
            else:
                b_lep_cand = b_tagged_jets[0]

            leptonic_top, lepton = self._leptonic_top_reconstruction(partons, iso_lepton, missing_et, b_lep_cand)
            if not lepton:
                continue

            lep_charge = iso_lepton.Charge

            self._build_output(hadronic_top, hadronic_top_subjets, leptonic_top, lepton, lep_charge, train_data)

    def change_clustering(self):
        self.pnet_clustering = not self.pnet_clustering
        print(f"Changed clustering strategy. Now using {'ParticleNet' if self.pnet_clustering else 'CA'} declustering.")

    def inspect_fails(self):

        events = self.NEvents
        print(f"Percentage of events cut due to missing fatjet: {100 * self.NFatjetFail / events:.2f}%")
        events -= self.NFatjetFail
        print(f"Percentage of events cut due to not having exactly one isolated lepton: {100 * self.NLepFail / events:.2f}%")
        events -= self.NLepFail
        print(f"Percentage of events cut due to missing b-tagged jet: {100 * self.NBTagFail / events:.2f}%")
        events -= self.NBTagFail
        print(f"Percentage of events cut due to kinematic cuts: {100 * self.KinematicsFail / events:.2f}%")
        events -= self.KinematicsFail
        print(f"Percentage of events cut due to unclear lepton affiliation: {100 * self.LeptonIdentificationFail / events:.2f}%")
        events -= self.LeptonIdentificationFail
        print(f"Percentage of events cut due to unclear b quark affiliation: {100 * self.BIdentificationFail / events:.2f}%")
        events -= self.BIdentificationFail
        print(f"Percentage of events cut due to declustering: {100 * self.DeclusteringFail / events:.2f}%")
        events -= self.DeclusteringFail
        print(f"Percentage of events cut due to hadronic top mass reconstruction: {100 * self.TopMassFailHad / events:.2f}%")
        events -= self.TopMassFailHad
        print(f"Percentage of events cut due to parton jet matching: {100 * self.JetMatchFail / events:.2f}%")
        events -= self.JetMatchFail
        print(f"Percentage of events cut due to top mass after matching: {100 * self.FailFail / events:.2f}%")
        events -= self.FailFail
        print(f"Percentage of events cut due to leptonic top mass reconstruction: {100 * self.TopMassFailLep / events:.2f}%")
        events -= self.TopMassFailLep
        if self.ttH:
            print(f"Percentage of events cut due to missing photons: {100 * self.NPhotonFail / events:.2f}%")
            events -= self.NPhotonFail
            print(f"Percentage of events cut due to photon pT/eta cuts: {100 * self.NPhotonPtEtaFail / events:.2f}%")
            events -= self.NPhotonPtEtaFail
            print(f"Percentage of events cut due to photon ISR cuts: {100 * self.NPhotonISRFail / events:.2f}%")
            events -= self.NPhotonISRFail
            print(f"Percentage of events cut due to m_gamma_gamma cuts: {100 * self.NPhotonMgamgamFail / events:.2f}%")
            events -= self.NPhotonMgamgamFail
        print(f"Total events processed: {self.NEvents}")
        print(f"Surviving number of events: {events}")

    def read_delphes(self, filepath, train_data=True, detect_missing=False):

        files_delphes, files_parton = self._load_files(filepath)

        if detect_missing:
            self._detect_missing(files_delphes)

        print(f"Found {len(files_delphes)} root files to process.")

        for f_d, f_p in tqdm(zip(files_delphes, files_parton), total=len(files_delphes), desc="Processing Delphes files"):
            self._get_single_delphes(f_d, f_p, train_data)

        return self.jet_data.snapshot()