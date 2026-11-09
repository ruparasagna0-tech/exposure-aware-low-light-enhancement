# experiments/: research notes and small reproducibility files

This folder holds the plans, results and findings of each stage (Markdown), the scene-disjoint split files (`generalized_lol_v2/splits/`), the dataset-audit code and reports (`dataset_audit/`), and the Stage-2 training history (`phase3_model_selection/stage0/source_phase2/history.json`, used as the Stage-3 timing reference).

The notes are the original working records and are kept unchanged. They sometimes refer to material that is **not part of the public repository**:

| Referenced in the notes | Where it actually lives | How to obtain it |
|---|---|---|
| `…/checkpoints/`, `…/tierA_runs_local/…`, `best.pth` | Authors' Google Drive and local machine | Re-train (see `docs/WEIGHTS.md`) |
| `…/pilots/pilot_…/`, `…/runs/…`, `…/diagnostic/DIAG_REPORT.md`, `logs/` | Run folders written to Google Drive by the Colab notebooks | Produced when the notebooks are run |
| `*_file_md5.json` snapshots, `provenance.json`, `run_info.json` | Local integrity records (they contain machine-specific paths) | Not needed to run the code; the related tests skip without them |
| `outputs/`, `panels/`, montages, slide decks | Generated images | Produced by the scripts |
| Files marked as planned or proposed (for example `src/losses/colour.py` in `phase3_model_selection/RESEARCH_PLAN.md`) | Never created; the plan was superseded | — |
| Upstream files such as `configs/LOLv1.yml`, `train.py`, `data/options.py` in `phase3_model_selection/STAGE1_PLAN.md` | The official third-party repositories, not this one | See `third_party_src/DOWNLOAD_MANIFEST.md` |

All numbers quoted in the top-level README (and in the project report, which is not part of this repository) can be traced to these notes.
