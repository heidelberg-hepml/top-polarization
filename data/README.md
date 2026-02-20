# Data generation

Data was generated using the following commands:

generate p p > t{L} t~{L} / generate p p > t{R} t~{R} / generate p p > t t~

The corresponding run_card, madspin_card, pythia_card, delphes_card are provided.

## MadGraph Bugfix:

Encountered the following issue when merging the split Pythia files after showering
cat: write error: Argument list too long

The error was fixed by modifying the corresponding file in the madevent_interface.py file

The issue has already been adressed here https://bugs.launchpad.net/mg5amcnlo/+bug/2130574
And the corresponding fix is here https://github.com/mg5amcnlo/mg5amcnlo/commit/f2f48bf9ff37d7e36b0191be158d7b6fc0d6826e


# Data structure

The data is saved in the form of an awkward array and can be loaded with ak.from_parquet(data)

The two top quarks are reconstructed from the detector level objects. 

The training data hadronic subjets are matched to the respective parton-level objects via their PID (b-quark, u-quark, d-quark)

The following cuts are applied to the detector level events:

        - Exactly one isolated lepton
        - At least two b-tagged jets
        - At least one FatJet with pT > 250 GeV and eta < 3

Reconstructed top quarks are required to have a mass within a pre-specified mass window

The saved data has the following structure:

AWKWARD ARRAY

        - TopLep (Reconstructed leptonically decaying top quark)
                - pt
                - eta
                - phi
                - mass
        - Lepton 
                - pt
                - eta
                - phi
                - charge
        - TopHad (Reconstructed hadronically decaying top quark)
                - pt
                - eta
                - phi
                - mass
        -Subjets (From TopHad)
                - pt
                - eta
                - phi
                - mass
                - tag (5: b-quark, 2: u-quark, 1: d-quark, 0: not tagged)
                - Constituents
                        - pt
                        - eta
                        - phi
                        - mass
                        - charge
                        - pid
