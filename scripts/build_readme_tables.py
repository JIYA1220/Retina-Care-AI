"""
build_readme_tables.py
Copies the measured numbers from results/*.json|csv into README.md between the
<!-- RESULTS:START --> and <!-- RESULTS:END --> markers, so the README tables are
generated from the result files and never typed by hand.
"""

import json
import os
import re

import pandas as pd

import _common as C

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build() -> str:
    parts = []
    mp = os.path.join(C.RESULTS, "val_metrics.json")
    if not os.path.exists(mp):
        return "Not yet measured. Run `python scripts/evaluate.py`."
    m = json.load(open(mp))
    n = next(iter(m["variants"].values()))["n_images"]
    parts.append(f"### Validation-split performance (n = {n})\n")
    parts.append(md_table(
        ["Variant", "QWK", "95% CI (bootstrap, 1000 resamples)", "Accuracy"],
        [[k, f"{v['qwk']:.4f}", f"{v['qwk_ci95'][0]:.4f} to {v['qwk_ci95'][1]:.4f}",
          f"{v['accuracy'] * 100:.1f}%"] for k, v in m["variants"].items()]))
    parts.append(f"\nMeasured {m['date']} at commit `{m['git']['commit'][:8]}`. Default model "
                 f"(`config.yaml`): `{m['primary_model']}`. Per-grade metrics and confusion matrices: "
                 "`results/val_metrics.json`, `results/confusion_matrix*.png`. "
                 "Error breakdown: [results/ERROR_ANALYSIS.md](results/ERROR_ANALYSIS.md).")

    cp = os.path.join(C.RESULTS, "calibration.json")
    parts.append("\n### Calibration (15-bin ECE, NLL)\n")
    if os.path.exists(cp):
        c = json.load(open(cp))
        parts.append(md_table(
            ["Model", "ECE before", "ECE after (cross-fitted T)", "NLL before", "NLL after (cross-fitted T)",
             "ECE with shipped T (in-sample)"],
            [[k, f"{v['ece_before']:.4f}", f"{v['ece_after_crossfit']:.4f}", f"{v['nll_before']:.4f}",
              f"{v['nll_after_crossfit']:.4f}", f"{v['ece_shipped_insample']:.4f}"]
             for k, v in c["models"].items()]))
        parts.append("\nT is fitted on one half of the validation split and evaluated on the other half "
                     "(2-fold cross-fitting, seed 0). Cross-fitted temperatures: "
                     + ", ".join(f"{k} {v[0]:.3f} / {v[1]:.3f}" for k, v in c["crossfit_temperatures"].items())
                     + ". Shipped temperatures (`model/*_temp.pt`): "
                     + ", ".join(f"{k} {v:.4f}" for k, v in c["shipped_temperatures"].items())
                     + ". Plot: `results/reliability_diagrams.png`.")
    else:
        parts.append("Not yet measured.")

    rp = os.path.join(C.RESULTS, "robustness.csv")
    parts.append("\n### Robustness (QWK under perturbation, validation split, no retraining)\n")
    if os.path.exists(rp):
        r = pd.read_csv(rp)
        piv = r.pivot_table(index=["perturbation", "severity", "parameter"], columns="model", values="qwk",
                            sort=False).reset_index()
        parts.append(md_table(
            ["Perturbation", "Severity", "Parameter", "EfficientNet-B0 QWK", "Ensemble QWK"],
            [[row.perturbation, row.severity, row.parameter, f"{row.efficientnet_b0:.4f}",
              f"{row.ensemble:.4f}"] for row in piv.itertuples()]))
        worst = r[r.perturbation != "clean"].sort_values("qwk").groupby("model").head(1)
        parts.append("\nWorst case per model: " + "; ".join(
            f"{w.model} {w.qwk:.4f} ({w.perturbation}, {w.parameter})" for w in worst.itertuples())
            + ". Perturbations are applied to raw images downscaled to a 640 px long side "
              "(see `scripts/robustness.py`), so the clean row can differ slightly from the table above. "
              "Plot: `results/robustness.png`.")
    else:
        parts.append("Not yet measured.")
    return "\n".join(parts)


def main():
    path = os.path.join(C.ROOT, "README.md")
    text = open(path, encoding="utf-8").read()
    if START not in text or END not in text:
        raise SystemExit("README.md has no RESULTS markers")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END),
                 lambda _: f"{START}\n{build()}\n{END}", text, flags=re.S)
    open(path, "w", encoding="utf-8").write(new)
    print("README results tables updated")


if __name__ == "__main__":
    main()
