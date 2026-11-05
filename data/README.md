# MC generation and analysis

Pipeline that produces the training/test parquet files consumed by the
`experiments/` package, starting from MadGraph + Pythia + Delphes.

## Layout

```
data/
├── cards/             # MG/MadSpin/Delphes/param cards
├── models/            # UFO models loaded by MadGraph
│                      # (HC_NLO_X0_UFO for ttH, SMEFTsim_topU3l_alphaScheme_UFO for tt_ctg)
├── pipeline/          # Python: Delphes -> parquet
│   ├── config.py      # ProcessConfig (tt, tth)
│   ├── kinematics.py  # deltaR, phi/eta, inv_mass, delphes_to_fj
│   ├── declustering.py# pnet / CA strategies
│   ├── matching.py    # parton-jet and Delphes-b-jet matching
│   ├── reco.py        # hadronic / leptonic top reconstruction
│   ├── writer.py      # awkward ArrayBuilder
│   ├── cuts.py        # event-level cut thresholds + FailCounter
│   ├── reader.py      # DelphesReader orchestrator
│   └── cli.py         # `python -m data.pipeline`
├── postprocess/
│   └── concat.py      # `python -m data.postprocess.concat`
├── slurm/
│   ├── _env.sh        # shared env + helpers (sourced)
│   ├── _lib.sh        # generation + analysis shell helpers (sourced)
│   ├── worker_generate.sh
│   ├── worker_analyse.sh
│   ├── worker_gen_ana.sh
│   └── submit.sh      # user-facing submitter
├── samples/              # raw .root + .lhe.gz (tt)         [gitignored]
├── samples_ttH/          # raw .root + .lhe.gz (tth)        [gitignored]
├── samples_awk/          # per-run parquet  (tt)            [gitignored]
├── samples_ttH_awk/      # per-run parquet  (tth)           [gitignored]
├── samples_awk_comb/     # combined parquet (tt)            [gitignored]
├── samples_ttH_awk_comb/ # combined parquet (tth)           [gitignored]
├── samples_tt_ctg{,_awk,_awk_comb}/      # tt_ctg     (quadratic) [gitignored]
└── samples_tt_ctg_lin{,_awk,_awk_comb}/  # tt_ctg_lin (linear)    [gitignored]
```

## Configuration

The shell scripts read these env vars (defaults shown below):

| Variable        | Default                                                    |
|-----------------|------------------------------------------------------------|
| `DATA_DIR`      | `<repo>/data` (derived from `_env.sh` script location)     |
| `MG_BIN`        | `${DATA_DIR}/MG/bin/mg5_aMC`                               |
| `CARDS_DIR`     | `${DATA_DIR}/cards`                                        |
| `MODELS_DIR`    | `${DATA_DIR}/models`                                       |
| `CONDA_SETUP`   | `${HOME}/miniconda3/etc/profile.d/conda.sh`                |
| `CONDA_ENV_GEN` | `base`         (MadGraph + Pythia + Delphes)               |
| `CONDA_ENV_ANA` | `rootpy`       (ROOT + fastjet + awkward + pylorentz)      |
| `SCRATCH_BASE`  | `${DATA_DIR}/scratch`                                      |
| `MASS_WINDOWS`  | `"165:190 120:220"` (space-separated `LO:HI` pairs)        |

## Processes

| Process  | Generation                              | Label format         |
|----------|-----------------------------------------|----------------------|
| `tt`     | SM `p p > t t~` (polarized)             | `0`, `L`, `R`, `LR`, `RL` |
| `tth`    | `HC_NLO_X0_UFO`, `p p > t t~ x0`        | `0`, `45`, `-45` (alpha)  |
| `tt_ctg` | SMEFTsim topU3l alpha, `p p > t t~ NP<=1`   | `re,im` complex ctG (e.g. `0.5,0.3`) |
| `tt_ctg_lin` | SMEFTsim topU3l alpha, `p p > t t~ NP^2<=1` | `re,im` complex ctG (e.g. `0.5,0.3`) |

`semilep` picks the MadSpin decay card (`1|2`); `nrun` is the per-job
random-seed index.

For `tt_ctg` and `tt_ctg_lin`, the param card is built per-job by `prepare_smeft_param_card`
in `data/slurm/_lib.sh`: the SMEFTsim default `param_card_massless.dat` is
copied to scratch, the `ctGRe` value (Block SMEFT, lhacode 15) is patched
in via `sed`, and a `Block SMEFTcpv` with `ctGIm` (lhacode 8) is appended.

## Generating data

Generate only (writes `.root` + `.lhe.gz` to `samples*/`):

```bash
data/slurm/submit.sh --process tt --mode gen --range 1:40
```

Analyse only (reads `.root`+`.lhe.gz`, writes parquet to `samples*_awk/`):

```bash
data/slurm/submit.sh --process tt --mode ana --range 1:40 --skip-missing
```

Combined (generate -> analyse -> delete intermediate root/lhe):

```bash
data/slurm/submit.sh --process tth --mode gen+ana --range 1:600
```

Wave-throttled submission (each wave waits on the previous via `--dependency=afterok`):

```bash
data/slurm/submit.sh --process tth --mode gen+ana --range 1:600 --wave-size 10
```

Subset of labels or semileps:

```bash
data/slurm/submit.sh --process tt  --mode gen+ana --range 1:40 --labels 0 --semileps 1
data/slurm/submit.sh --process tth --mode gen+ana --range 1:600 --labels "0 45"
```

tt_ctg / tt_ctg_lin (SMEFT) — `--labels` is required and takes one or more
`re,im` points:

```bash
data/slurm/submit.sh --process tt_ctg     --mode gen+ana --range 1:10 \
    --labels "0.5,0.3 1.0,0.0 0,-0.5"
data/slurm/submit.sh --process tt_ctg_lin --mode gen+ana --range 1:10 \
    --labels "0.5,0.3 1.0,0.0 0,-0.5"
```

Each label produces files tagged `<process>_re_<R>_im_<I>_semilep_<S>_nrun_<N>`
(decimal points replaced with `p`, e.g. `tt_ctg_lin` + `0.5,0.3` ->
`tt_ctg_lin_re_0p5_im_0p3_...`), so the two truncations never collide.

## EFT truncation: `tt_ctg` vs `tt_ctg_lin`

Both processes use the same model, cards and reconstruction; they differ only in
the coupling-order constraint on the `generate` line
(`data/slurm/_env.sh::emit_mg_cmds`).

`NP<=1` constrains the **amplitude** to at most one dim-6 insertion,
`M = M_SM + M_ctG`. MadGraph squares the whole amplitude, so `|M|^2` retains
three pieces:

| term                    | NP power in `\|M\|^2` | `tt_ctg` (`NP<=1`) | `tt_ctg_lin` (`NP^2<=1`) |
|-------------------------|------------------------|--------------------|--------------------------|
| `\|M_SM\|^2`              | 0                      | kept               | kept                     |
| `2Re(M_SM* M_ctG)`      | 1                      | kept               | kept                     |
| `\|M_ctG\|^2` (dim-6 sq.) | 2                      | kept               | **dropped**              |

`NP^2<=1` is a **squared**-order constraint, so `tt_ctg_lin` gives
`sigma = sigma_SM + c * sigma_int`, strictly linear in the Wilson coefficient.
You can see the selection in the generated matrix element — `SQSPLITORDERS`
lists the NP^2 = 0, 1, 2 buckets and `CHOSEN_SO_CONFIGS` is
`.TRUE., .TRUE., .FALSE.`

Two consequences worth knowing:

1. **MadSpin cannot consume a squared-order process line, so the recorded
   process is rewritten around it.** MadSpin rebuilds its matrix elements from
   the `generate` line stored in `Cards/proc_card_mg5.dat`, turning it into a
   decay chain (`generate p p > t t~ NP^2<=1, (t > b w+, w+ > j j), ...`), and
   MG5 rejects that with *"Decay processes cannot specify squared orders
   constraints"* — the job dies after the production run. `emit_mg_cmds`
   therefore emits a `shell sed` that rewrites the recorded line to `NP<=1`
   *after* `output` and *before* `launch`. The production matrix element has
   already been exported at that point and keeps its truncation
   (`CHOSEN_SO_CONFIGS = .TRUE.,.TRUE.,.FALSE.`), so the generated events remain
   linear in ctG; only MadSpin's decay and spin-correlation matrix elements use
   the full `|M|^2`. `restore_linear_proc_card` (in `_lib.sh`) puts `NP^2<=1`
   back into the shipped LHE banner once MadSpin has finished, so the sample
   documents what was actually generated.

   Physics consequence: **the truncation is exact for the production cross
   section and kinematics, but the decay angular correlations carry a residual
   O(ctG^2)**. That matters here, since polarization is read off exactly those
   angles. It is a bounded inconsistency — the same order as the term being
   dropped — but it is not removable with MadSpin as it currently stands.

2. **Negative weights.** A linear-truncated `|M|^2` is not positive definite;
   where the ctG interference overwhelms the SM term the weight flips sign.
   MadEvent unweights on `abs(wgt)` and keeps the sign, so such events reach the
   LHE with a negative weight — and the parquet pipeline does not read weights,
   it counts every event as +1. `run_generation` therefore calls
   `report_negative_weights` for `tt_ctg_lin` and prints the fraction to the job
   log. **Check it before trusting a sample.** Measured at 13.6 TeV with
   `run_card_top.dat`:

   | ctGRe | sigma (pb) | negative-weight fraction |
   |-------|-----------|--------------------------|
   | 0     | 14.09     | 0 %                      |
   | +1    | 10.41     | 2.9 %                    |
   | -5    | 33.27     | 0 %                      |
   | +5    | -4.71     | 93.2 %                   |

   `ctGRe = +5` gives a *negative* cross section: far outside the range where a
   linear truncation means anything. A large negative fraction is the signal to
   back off to smaller `|ctG|`.

## Validating the truncation

The interference term is odd in the Wilson coefficient, so a linear-only sample
obeys `sigma(+c) + sigma(-c) = 2 * sigma_SM` exactly, while a quadratic one
overshoots by `2 c^2 sigma_quad`. With the numbers above:

- `tt_ctg_lin`: `-4.71 + 33.27 = 28.56` vs `2 * 14.09 = 28.18` (1.3 %, MC noise)
- `tt_ctg`:     `21.41 + 58.94 = 80.35`, i.e. `sigma_quad = 1.04 pb`, which
  reproduces both quadratic points to better than 1 %.

To repeat it, generate the same `|ctG|` with both signs plus the `0,0` point and
compare the `Integrated weight (pb)` values in the LHE banners.

## Combining shards

```bash
python -m data.postprocess.concat --process tt  --input-dir data/samples_awk \
    --output-dir data/samples_awk_comb --range 1:69

python -m data.postprocess.concat --process tth --input-dir data/samples_ttH_awk \
    --output-dir data/samples_ttH_awk_comb --range 1:600

python -m data.postprocess.concat --process tt_ctg_lin \
    --input-dir data/samples_tt_ctg_lin_awk \
    --output-dir data/samples_tt_ctg_lin_awk_comb --range 1:50 --labels 0.5,0.3
```

## UFO models

UFO models are tracked in the repo so runs are reproducible without
dropping anything into MadGraph's own `models/` directory. `import model`
lines in the MG command file (see `data/slurm/_env.sh::emit_mg_cmds`)
resolve via `${MODELS_DIR}`.

| Process  | Model                                                  | Source                                              |
|----------|--------------------------------------------------------|-----------------------------------------------------|
| `tth`    | `data/models/HC_NLO_X0_UFO`                            | FeynRules Higgs Characterisation                    |
| `tt_ctg`, `tt_ctg_lin` | `data/models/SMEFTsim_topU3l_alphaScheme_UFO` | SMEFTsim v3.0.2 (`topU3l_alphaScheme`, MFV-free) |

The SMEFTsim topU3l alpha scheme exposes `ctGRe` (Block SMEFT, code 15)
and `ctGIm` (Block SMEFTcpv, code 8) as independent external parameters;
internally MadGraph combines them as `ctG = ctGRe + i*ctGIm`.

To use a different model location, export `MODELS_DIR` before submitting.

## Data structure

Each parquet event has the following schema (see `data/pipeline/writer.py`):

- `TopLep`: `(pt, eta, phi, mass)` — reconstructed leptonic top.
- `Lepton`: `(pt, eta, phi, charge)`.
- `TopHad`: `(pt, eta, phi, mass)` — reconstructed hadronic top.
- `Subjets[]` (from `TopHad`): `(pt, eta, phi, mass, tag, Constituents[])`.
  `tag` is the signed quark id when training data (5: b, ±2/±4: up-type,
  ±1/±3: down-type); 5 for the matched b-jet and 0 for the rest when test data.
  Each `Constituent` has `(pt, eta, phi, mass, charge, pid)`.
- ttH additionally writes `photon_leading`, `photon_subleading`, and `higgs`
  (di-photon reconstruction), each `(pt, eta, phi, mass)`.

## Cuts (detector level)

- `>= 1` FatJet with `pT > 250 GeV`, `|eta| < 3` (leading).
- Exactly one isolated lepton.
- `>= 2` Delphes b-tagged jets.
- Reconstructed top mass within window (`165 - 190 GeV` strict, `120 - 220 GeV` loose).
- ttH: `>= 2` photons with `pT > 20 GeV`, `|eta| < 2.5`, `pT_lead/m_gg > 0.35`,
  `pT_sub/m_gg > 0.25`, `110 < m_gg < 140 GeV`.

## MadGraph bugfix

When merging Pythia files after showering, MadGraph 5 may fail with
`cat: write error: Argument list too long`. Reported and patched upstream:

- Issue: https://bugs.launchpad.net/mg5amcnlo/+bug/2130574
- Fix: https://github.com/mg5amcnlo/mg5amcnlo/commit/f2f48bf9ff37d7e36b0191be158d7b6fc0d6826e

Apply the patch to the local `madevent_interface.py` before generating.
