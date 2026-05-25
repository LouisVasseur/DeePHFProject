# AI Use: code mostly made by Claude based on the rest of the codebase, with some hand edits for file handling and robustness.

"""Read the JSONs in results/ and write figures + LaTeX tables.

Missing files are skipped with a notice; the rest still render.
"""

import argparse
import json
from pathlib import Path

import figures
import tables

# load: helper to load a JSON file, returning None if not found (with a notice).
def load(path):
    p = Path(path)
    if not p.exists():
        print(f"  skip: {path} not found")
        return None
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--tables-dir", default="tables")
    args = p.parse_args()

    R = Path(args.results_dir)
    fig_d = Path(args.figures_dir); fig_d.mkdir(parents=True, exist_ok=True)
    tab_d = Path(args.tables_dir);  tab_d.mkdir(parents=True, exist_ok=True)

    grid = load(R / "results.json")
    ab   = load(R / "edge_bias_ablation.json")
    ef   = load(R / "edge_features_ablation.json")
    ks   = load(R / "k_sweep.json")
    rot  = load(R / "rotation_invariance.json")

    if grid: figures.fig_in_domain(grid, fig_d / "fig1_in_domain.pdf")
    if grid: figures.fig_feature_scaling(grid, fig_d / "fig2_feature_scaling.pdf")
    if ab:   figures.fig_edge_bias(ab, fig_d / "fig3_edge_bias.pdf")
    if ef:   figures.fig_edge_features(ef, fig_d / "fig4_edge_features.pdf")
    if ks:   figures.fig_k_sweep(ks, fig_d / "fig5_k_sweep.pdf")

    if grid: tables.table_best_cells(grid, tab_d / "table1_best_cells.tex")
    if grid: tables.table_seed_variance(grid, tab_d / "table2_seed_variance.tex")

    if rot is not None:
        print(f"\nRotation invariance: rel_dev_max = {rot.get('rel_dev_max', '?'):.2e}"
              f"  verdict = {rot.get('verdict', '?')}")


if __name__ == "__main__":
    main()
