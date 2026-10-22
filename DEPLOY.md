# Deployment guide (manual steps)

Nothing here has been run or tested; no deployment has been done. Do these steps yourself.

## What the app needs at runtime

| Item | Size | Note |
|---|---|---|
| `model/efficientnet_b0_dr.pth` | 17 MB | default model |
| `model/resnet50_dr.pth` | 92 MB | only for ensemble mode |
| `model/efficientnet_b0_temp.pt`, `model/resnet50_temp.pt` | ~1 KB each | temperatures |
| `results/` | about 1 MB | the Evaluation page reads only this folder |
| `data/train_images/` | not in the repo | **not needed**: the Evaluation page works from `results/` |

`model/efficientnet_dr.pth` (17 MB) is an older checkpoint that nothing in the code loads. It can be removed from the repo to save space (not done; deleting needs your OK).

## Option A: Streamlit Community Cloud

1. Merge `fix/real-evaluation` into `main` (after you have reviewed it) and push.
2. Go to https://share.streamlit.io, choose "New app", pick the repo `JIYA1220/Retina-Care-AI`, branch `main`, main file `app.py`.
3. In "Advanced settings" choose **Python 3.10 or 3.11**. `requirements.txt` pins `torch==2.1.0` and `numpy==1.24.4`, which do not install on newer Python versions.
4. Deploy. `packages.txt` already lists the system library (`libgl1`) that OpenCV needs.
5. The `resnet50_dr.pth` file is 92 MB. GitHub accepts files under 100 MB (with a warning above 50 MB). If it ever grows, use Git LFS.
6. Free-tier memory is limited. Loading both models plus LIME / SHAP may exceed it. If the app crashes, turn off ensemble mode or the SHAP tab first. This has not been tested.

## Option B: Hugging Face Spaces

1. Create a Space at https://huggingface.co/new-space. Check the current SDK options in the Hugging Face docs; if a native Streamlit SDK is offered use it, otherwise use the Docker SDK with a Dockerfile that installs `requirements.txt` and runs `streamlit run app.py --server.port 7860`.
2. Spaces require Git LFS or Xet for files over 10 MB, so run `git lfs track "*.pth"` before pushing the checkpoints.
3. Add the Space as a second git remote and push the `main` branch, or upload the files in the web UI.

## Before you deploy

- Run `pytest tests` and `streamlit run app.py` locally and open all three pages.
- Do not set `DR_DEMO_MOCK` in the deployed environment.
- Re-run `python scripts/evaluate.py` if the checkpoints change, and commit the new `results/` files, so the Evaluation page matches the deployed weights.
