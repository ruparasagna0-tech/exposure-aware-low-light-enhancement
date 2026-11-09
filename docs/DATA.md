# Datasets

No dataset is included in this repository. Download each one from its official page, accept its terms (research use only), and place the archives exactly as shown below. All preparation scripts read the archives in place, never modify them, and record an MD5 for every file they write.

## Official sources

Pages checked 2026-11-09. Each page loaded and is the authors' official page. The individual Google Drive, Baidu and Kaggle files linked from them were not downloaded during this check.

| Dataset | Official page | Download links listed on that page |
|---|---|---|
| **LOL (LOL-v1)**, Wei et al., BMVC 2018 | https://daooshee.github.io/BMVC2018website/ | "LOw Light paired dataset (LOL)": Google Drive and Baidu Pan (code `acp3`) |
| **LOL-v2**, Yang et al., IEEE TIP 2021 | https://github.com/flyywh/CVPR-2020-Semi-Low-Light (item "LOL-v2 (the extension work)") | Baiduyun (extraction code `l9xm`) and Google Drive. The authors' TIP repository https://github.com/flyywh/SGM-Low-Light links only result archives, not the dataset |
| **SICE**, Cai, Gu, Zhang, IEEE TIP 2018 | https://github.com/csjcai/SICE | "Part1: 360 Image Sequences" and "Part2: 229 Image Sequences" via Google Drive or BaiduYun (Part 2 data and labels are separate Baidu links) |
| **LoLI-Street**, Islam et al., ACCV 2024 | https://github.com/tanvirnwu/TriFuse | Kaggle: https://www.kaggle.com/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street, or Google Drive |

## Exact archive names and layout expected by the scripts

Paths are relative to the repository root. Rename a downloaded archive if its name differs.

| Path | Contents (top level inside the archive) | Read by |
|---|---|---|
| `data/LOL/LOLdataset.zip`, extracted to `data/LOL/LOLdataset/` | `our485/{low,high}/`, `eval15/{low,high}/` | Stage-1 baseline only; `LOL_V1_ROOT` can point elsewhere |
| `data/LOL-v2/LOL-v2.zip` | `LOL-v2/Real_captured/{Train,Test}/{Low,Normal}/`, `LOL-v2/Synthetic/{Train,Test}/{Low,Normal}/` | `prepare_phase2_data.py` |
| `data/SICE/Dataset_Part1.rar` | `Dataset_Part1/<id>/<k>.JPG`, `Dataset_Part1/Label/<id>.JPG` | `prepare_phase2_data.py`, `prepare_phase5_diag_data.py` |
| `data/SICE/Dataset_Part2.rar` | `Dataset_Part2/...`, `testing index.txt` | only the text file `testing index.txt` (see below) |
| `data/LoLI-Street/<any name>.zip` (exactly one zip; Kaggle names it `archive.zip`) | `LoLI-Street Dataset/{Train,Val,Test,YOLO Annotations}/...` | `prepare_phase2_data.py`, `prepare_phase5_diag_data.py` |

If the Baidu Part 2 download arrives as separate data and label archives, combine them into a single `Dataset_Part2.rar` with the layout above. Only `testing index.txt` is needed for the experiments in this repository.

`data/` and `data_prepared/` are git-ignored.

## Preparation

```bash
python scripts/prepare_phase2_data.py --zip        # LOL-v2 + SICE test sequences + LoLI-Street Val -> data_prepared/phase2 (+ .zip)
python scripts/make_phase2_splits.py               # regenerates and verifies experiments/generalized_lol_v2/splits/*.json
python scripts/prepare_phase5_diag_data.py --tmp <scratch dir>   # SICE Part 1 (all exposures, long side 900) + LoLI-Street Val/Test
```

The scene-disjoint split files used in all reported experiments are included in `experiments/generalized_lol_v2/splits/`. `make_phase2_splits.py` regenerates them bit-identically from the data.

## Sealed data (SICE Part 2)

SICE Part 2 is reserved for a single final evaluation after the method and its settings are frozen.

- `scripts/prepare_phase5_diag_data.py` refuses any archive other than `Dataset_Part1.rar`.
- `scripts/phase5_diagnostic.py` refuses Part 2 paths.
- Transparency note: the Stage-2 preparation script (`prepare_phase2_data.py`, written before this rule) reads only the text file `testing index.txt` from `Dataset_Part2.rar`, which lists the official SICE Part 1 test sequences. It reads no Part 2 images.

## Other conventions

- LoLI-Street Val id 30727 is excluded: its image is completely black.
- LOL-v1 is excluded from all generalised experiments: all 500 of its pairs occur in LOL-v2 Real (`experiments/dataset_audit/overlap_audit.txt`).
- Mirrors and re-uploads may differ in content or naming. The preparation manifests (`data_prepared/*/manifest.json`) record the MD5 of every prepared file, so differences can be detected.
