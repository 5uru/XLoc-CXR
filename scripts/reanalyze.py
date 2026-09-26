"""Post-hoc statistical reanalysis of evaluation results (no dataset needed).

Outputs outputs/reanalysis.json:
  - per layer x method x metric: mean, SD, n, cluster-bootstrap 95% CI (by image)
  - paired Wilcoxon tests (same image+class): layer4 vs layer3, method vs method
  - Benjamini-Hochberg FDR correction across test families
  - per-class pointing game: n, mean, bootstrap CI
  - combined (PG+EB)/2 scores per class (clarifies controls' score definition)
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

SEED = 0
N_BOOT = 2000
OUT = "outputs/reanalysis.json"


def bh_fdr(pvals: list) -> list:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


def cluster_bootstrap_ci(df: pd.DataFrame, value_col: str,
                         rng: np.random.RandomState, n_boot: int = N_BOOT):
    """Bootstrap the mean, resampling whole images (cluster = image_idx)."""
    per_image = df.groupby("image_idx")[value_col].mean()
    vals = per_image.values
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    idx = rng.randint(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def paired_test(df: pd.DataFrame, col_a: str, col_b: str,
                higher_is_better: bool) -> dict:
    """Wilcoxon signed-rank on (image, class) pairs. a = candidate, b = baseline."""
    wide = df.pivot_table(index=["image_idx", "class"],
                          columns="variant", values="score")
    if col_a not in wide.columns or col_b not in wide.columns:
        return {"error": "missing variant"}
    wide = wide[[col_a, col_b]].dropna()
    a, b = wide[col_a].values, wide[col_b].values
    diff = a - b
    n_eff = int((diff != 0).sum())
    if n_eff < 5:
        # all (or nearly all) pairs tied -> no evidence of a difference
        return {"n_pairs": int(len(diff)), "n_nonzero": n_eff,
                "median_diff": float(np.median(diff)),
                "mean_diff": float(np.mean(diff)),
                "frac_a_better": 0.5, "ties": int((diff == 0).sum()),
                "p_value": 1.0}
    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
        if np.isnan(p):
            p = 1.0
    except ValueError:
        return {"n_pairs": int(len(diff)), "n_nonzero": n_eff,
                "p_value": 1.0}
    better = float((diff > 0).mean()) if higher_is_better else float((diff < 0).mean())
    return {
        "n_pairs": int(len(diff)),
        "n_nonzero": n_eff,
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
        "frac_a_better": better,
        "ties": int((diff == 0).sum()),
        "p_value": float(p),
    }


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_csv("outputs/results.csv")
    findings = df[df["class"] != "No finding"].copy()
    combined = df[df["class"] != "No finding"].copy()

    out = {"n_images": int(df.image_idx.nunique()), "seed": SEED,
           "n_boot": N_BOOT}

    # ---- 1. Descriptive: mean, SD, cluster-bootstrap CI ----
    desc = {}
    for (layer, method, metric), g in findings.groupby(["layer", "method", "metric"]):
        g = g.dropna(subset=["score"])
        ci = cluster_bootstrap_ci(g, "score", rng)
        desc.setdefault(layer, {}).setdefault(method, {})[metric] = {
            "mean": float(g["score"].mean()),
            "sd": float(g["score"].std(ddof=1)),
            "n": int(len(g)),
            "ci95": ci,
        }
    out["descriptive"] = desc

    # ---- 2. Paired tests: layer4 vs layer3 (per method) ----
    family = []
    for method in ["grad_cam", "grad_cam_plus_plus", "xgrad_cam"]:
        sub = findings[findings.method == method].copy()
        sub["variant"] = sub["layer"]
        for metric in ["pointing_game", "energy_box", "iou", "distance"]:
            m = sub[sub.metric == metric]
            r = paired_test(m, "layer4", "layer3",
                            higher_is_better=(metric != "distance"))
            family.append({"test": f"layer4_vs_layer3/{method}/{metric}", **r})
    pvals = [t["p_value"] for t in family]
    for t, q in zip(family, bh_fdr(pvals)):
        t["p_fdr"] = q
    out["tests_layer"] = family

    # ---- 3. Paired tests: methods at each layer ----
    family = []
    for layer in ["layer3", "layer4"]:
        sub = findings[findings.layer == layer].copy()
        sub["variant"] = sub["method"]
        for metric in ["pointing_game", "energy_box", "iou", "distance"]:
            m = sub[sub.metric == metric]
            for a, b in [("grad_cam_plus_plus", "grad_cam"),
                         ("xgrad_cam", "grad_cam"),
                         ("grad_cam_plus_plus", "xgrad_cam")]:
                r = paired_test(m, a, b,
                                higher_is_better=(metric != "distance"))
                family.append({"test": f"{a}_vs_{b}/{layer}/{metric}", **r})
    pvals = [t["p_value"] for t in family]
    for t, q in zip(family, bh_fdr(pvals)):
        t["p_fdr"] = q
    out["tests_method"] = family

    # ---- 4. Per-class pointing game (layer4, best method) ----
    pg = findings[(findings.layer == "layer4")
                  & (findings.method == "grad_cam_plus_plus")
                  & (findings.metric == "pointing_game")].dropna()
    per_class = {}
    for cls, g in pg.groupby("class"):
        ci = cluster_bootstrap_ci(g, "score", rng)
        per_class[cls] = {
            "n": int(len(g)),
            "mean": float(g["score"].mean()),
            "sd": float(g["score"].std(ddof=1)),
            "ci95": ci,
        }
    out["per_class_pg_layer4_gpp"] = per_class

    # ---- 5. Combined score (PG+EB)/2 per class, grad_cam layer4 (controls' definition) ----
    wide = (findings[(findings.layer == "layer4")
                     & (findings.method == "grad_cam")]
            .pivot_table(index=["image_idx", "class"],
                         columns="metric", values="score")
            .dropna(subset=["pointing_game", "energy_box"]))
    wide["combined"] = (wide["pointing_game"] + wide["energy_box"]) / 2
    comb = {}
    for cls, g in wide.groupby(level="class"):
        v = g["combined"].values
        comb[cls] = {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "frac_zero": float((v == 0).mean()),
            "sd": float(v.std(ddof=1)),
        }
    out["combined_score_gradcam_layer4"] = comb
    allv = wide["combined"].values
    out["combined_score_global"] = {
        "n": int(len(allv)), "mean": float(allv.mean()),
        "median": float(np.median(allv)),
        "frac_zero": float((allv == 0).mean()),
        "sd": float(allv.std(ddof=1)),
    }

    # ---- 6. Per-class PG: grad++ vs grad, layer4 (FDR over classes) ----
    wide2 = (findings[(findings.layer == "layer4")
                      & (findings.metric == "pointing_game")]
             .pivot_table(index=["image_idx", "class"],
                          columns="method", values="score").dropna())
    family = []
    for cls in wide2.index.get_level_values("class").unique():
        sub = wide2.xs(cls, level="class")
        if len(sub) < 5:
            continue
        try:
            _, p = stats.wilcoxon(sub["grad_cam_plus_plus"],
                                  sub["grad_cam"], alternative="greater")
        except ValueError:
            p = float("nan")
        family.append({"test": f"gpp_vs_grad_pg/{cls}",
                       "n_pairs": int(len(sub)), "p_value": float(p)})
    pvals = [t["p_value"] for t in family]
    for t, q in zip(family, bh_fdr(pvals)):
        t["p_fdr"] = q
    out["tests_per_class_pg"] = family

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    # ---- Console summary ----
    print(f"images={out['n_images']}")
    print("\n== Table layer4 (mean ± SD [CI]) ==")
    for metric in ["pointing_game", "energy_box", "iou", "distance"]:
        row = []
        for m in ["grad_cam", "grad_cam_plus_plus", "xgrad_cam"]:
            d = desc["layer4"][m][metric]
            row.append(f"{m[:6]}={d['mean']:.3f}±{d['sd']:.3f}"
                       f"[{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]")
        print(f"  {metric:15s}", "  ".join(row))
    print("\n== layer4 vs layer3 (Wilcoxon paired, BH-FDR) ==")
    for t in out["tests_layer"]:
        print(f"  {t['test']:55s} p={t['p_value']:.3g} q={t['p_fdr']:.3g} "
              f"med_diff={t.get('median_diff', float('nan')):+.4f}")
    print("\n== methods == (spot check gpp vs grad layer4)")
    for t in out["tests_method"]:
        if "layer4" in t["test"] and "pointing" in t["test"]:
            print(f"  {t['test']:50s} p={t['p_value']:.3g} q={t['p_fdr']:.3g}")
    print("\n== per-class PG layer4 gpp ==")
    for cls, d in sorted(per_class.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {cls:22s} n={d['n']:3d} {d['mean']:.3f} "
              f"[{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]")
    print("\n== combined (PG+EB)/2 grad layer4 ==")
    g = out["combined_score_global"]
    print(f"  global n={g['n']} mean={g['mean']:.4f} median={g['median']:.4f} "
          f"frac_zero={g['frac_zero']:.2%}")
    for cls in ["Cardiomegaly", "ILD"]:
        c = comb.get(cls)
        if c:
            print(f"  {cls}: mean={c['mean']:.4f} median={c['median']:.4f} "
                  f"frac_zero={c['frac_zero']:.1%} n={c['n']}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
