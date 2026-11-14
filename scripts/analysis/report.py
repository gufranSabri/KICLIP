#!/usr/bin/env python3
# CPU report over dump.py's outputs: accuracy decompositions, field strength / geometry / layer attribution, base pull
# in the joint label space, layer-wise separability and drift, text drift, calibration, per-class results, curves.
#
#     OMP_NUM_THREADS=1 python scripts/analysis/report.py --dumps runs/analysis/dumps \
#         --runs froster ki field_pool_dropout --out docs/analysis

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CLIP_NAME = "clip"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SPLIT_COLORS = {"base": "#4a3aa7", "novel": "#e34948"}
SPLIT_STYLES = {"base": "-", "novel": "--"}
ALPHAS = [i / 10 for i in range(21)]
LAMBDAS = [i / 10 for i in range(11)]
FOLDS = 5


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
        "legend.fontsize": 8, "text.color": INK, "font.family": "sans-serif",
    })


def hm(b: float, n: float) -> float:
    return 2 * b * n / (b + n) if b + n else 0.0


def view_logits(emb: torch.Tensor, classes: torch.Tensor, scale: float) -> torch.Tensor:  # (n, V, T, D) -> (n, V, C)
    return scale * torch.einsum("nvtd,cd->nvc", F.normalize(emb.float(), dim=-1), classes.float()) / emb.shape[2]


def top1(logits: torch.Tensor, y: torch.Tensor) -> float:  # (n, V, C) summed over views, as src/test.py
    return 100 * (logits.sum(1).argmax(-1) == y).float().mean().item()


def top5(logits: torch.Tensor, y: torch.Tensor) -> float:
    k = min(5, logits.shape[-1])
    return 100 * (logits.sum(1).topk(k, -1).indices == y[:, None]).any(-1).float().mean().item()


def video_mean(emb: torch.Tensor) -> torch.Tensor:  # (n, V, T, D) -> (n, D) mean of unit frames, renormalized
    return F.normalize(F.normalize(emb.float(), dim=-1).mean((1, 2)), dim=-1)


def field_apply(state: dict[str, torch.Tensor], pooled: torch.Tensor) -> torch.Tensor:  # (..., L, M) -> (..., D)
    x = F.gelu(F.linear(pooled.float().flatten(-2), state["net.1.weight"], state["net.1.bias"]))
    return F.linear(x, state["net.3.weight"], state["net.3.bias"])


def text_basis(classes: torch.Tensor) -> torch.Tensor:  # (C, D) -> orthonormal rows (C, D) spanning the class texts
    return torch.linalg.qr(classes.float().T).Q.T


def energy(x: torch.Tensor, basis: torch.Tensor) -> float:  # % of each unit row's squared norm inside span(basis)
    x = F.normalize(x.float().reshape(-1, x.shape[-1]), dim=-1)
    return 100 * (x @ basis.T).pow(2).sum(-1).mean().item()


def ncm(x: torch.Tensor, y: torch.Tensor) -> float:
    # nearest class mean, FOLDS-fold; folds are contiguous per class so clips cut from one source video stay together
    x = x.float()
    fold = torch.empty_like(y)
    for c in y.unique():
        idx = (y == c).nonzero().squeeze(1)
        fold[idx] = torch.arange(len(idx)) * FOLDS // len(idx)
    labels, correct = y.unique(), 0
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
        z = F.normalize((x - mu) / sd, dim=-1)
        means = F.normalize(torch.stack([z[tr & (y == c)].mean(0) for c in labels]), dim=-1)
        correct += (labels[(z[te] @ means.T).argmax(-1)] == y[te]).sum().item()
    return 100 * correct / len(y)


def cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float() - a.float().mean(0), b.float() - b.float().mean(0)
    return ((a.T @ b).norm() ** 2 / ((a.T @ a).norm() * (b.T @ b).norm())).item()


def erank(x: torch.Tensor) -> float:
    s = torch.linalg.svdvals(x - x.mean(0)).pow(2)
    p = s / s.sum()
    return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()


def fisher(x: torch.Tensor, y: torch.Tensor) -> float:  # between- over within-class scatter
    mu, sb, sw = x.mean(0), 0.0, 0.0
    for c in y.unique():
        xc = x[y == c]
        mc = xc.mean(0)
        sb += len(xc) * (mc - mu).pow(2).sum().item()
        sw += (xc - mc).pow(2).sum().item()
    return sb / sw


def calibration(logits: torch.Tensor, y: torch.Tensor, bins: int = 15) -> dict:
    p = logits.mean(1).softmax(-1)
    conf, pred = p.max(-1)
    right = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece, curve = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(right[m].mean().item() - conf[m].mean().item())
            curve.append((conf[m].mean().item(), 100 * right[m].mean().item()))
    two = p.topk(2, -1).values
    return {
        "acc": 100 * right.mean().item(), "conf": 100 * conf.mean().item(), "ece": 100 * ece,
        "margin": 100 * (two[:, 0] - two[:, 1]).mean().item(),
        "conf_right": 100 * conf[right.bool()].mean().item() if right.any() else 0.0,
        "conf_wrong": 100 * conf[~right.bool()].mean().item() if (~right.bool()).any() else 0.0,
        "curve": curve,
    }


def short(classname: str) -> str:
    return classname.split(":")[0]


class Run:
    def __init__(self, path: Path):
        d = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        self.name, self.checkpoint, self.config = d["name"], d["checkpoint"], d["config"]
        self.scale, self.alpha, self.segments = d["logit_scale"], d["field_alpha"], d["segments"]
        self.field_embed = d["field_embed"]
        self.splits = d["splits"]
        self.is_clip = self.checkpoint == CLIP_NAME
        self.color = MUTED
        self._cache: dict = {}

    @property
    def has_field(self) -> bool:
        return self.field_embed is not None

    @property
    def variants(self) -> list[str]:
        return ["final", "backbone", "field"] if self.has_field else ["final"]

    def run_dir(self) -> Path | None:
        if self.is_clip:
            return None
        path = Path(self.checkpoint)
        return (path if path.is_absolute() else REPO_ROOT / path).parent

    def y(self, split: str) -> torch.Tensor:
        return self.splits[split]["labels"]

    def classes(self, split: str) -> torch.Tensor:
        return self.splits[split]["classes"].float()

    def emb(self, split: str, variant: str = "final", alpha: float | None = None) -> torch.Tensor:
        s = self.splits[split]
        if variant == "field":
            return s["field"].float()
        if variant == "backbone" or not self.has_field:
            return s["head"].float()
        return s["head"].float() + (self.alpha if alpha is None else alpha) * s["field"].float()

    def logits(self, split: str, variant: str = "final") -> torch.Tensor:  # (n, V, C), cached
        key = (split, variant)
        if key not in self._cache:
            self._cache[key] = view_logits(self.emb(split, variant), self.classes(split), self.scale)
        return self._cache[key]


class Report:
    def __init__(self, out: Path):
        self.out, self.lines, self.metrics = out, [], {}
        (out / "figures").mkdir(parents=True, exist_ok=True)

    def section(self, title: str, note: str) -> None:
        self.lines += ["", f"## {title}", "", note, ""]

    def text(self, line: str) -> None:
        self.lines += [line, ""]

    def table(self, header: list[str], rows: list[list]) -> None:
        cell = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        self.lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        self.lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
        self.lines.append("")

    def figure(self, fig, name: str, caption: str) -> None:
        for ext in (".png", ".pdf"):
            fig.savefig(self.out / "figures" / f"{name}{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        self.lines += [f"![{caption}](figures/{name}.png)", ""]

    def write(self) -> None:
        (self.out / "report.md").write_text("\n".join(self.lines).strip() + "\n")
        (self.out / "metrics.json").write_text(json.dumps(self.metrics, indent=2))


def legend_runs(ax, runs: list[Run], styles: bool = False) -> None:
    handles = [plt.Line2D([], [], color=r.color, lw=2, label=r.name) for r in runs]
    if styles:
        handles += [plt.Line2D([], [], color=INK2, ls=SPLIT_STYLES[k], lw=1.4, label=k) for k in SPLIT_STYLES]
    ax.legend(handles=handles, loc="best")


# ---- sections ---------------------------------------------------------------------------------------------------


def sanity(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("0. Sanity", "Top-1 recomputed from the dumped fp16 embeddings against each run's results.json; "
                "for field runs, cosine between the dumped field and field_embed re-applied to the dumped pooled input.")
    rows = []
    for r in runs:
        for key in ("base", "novel"):
            ours = top1(r.logits(sp[key]), r.y(sp[key]))
            ref = "-"
            if r.run_dir() is not None:
                path = r.run_dir() / "results.json"
                if path.exists():
                    ref = json.loads(path.read_text()).get(key, {}).get("top1", "-")
            recompute = "-"
            if r.has_field:
                s = r.splits[sp[key]]
                recompute = F.cosine_similarity(field_apply(r.field_embed, s["pooled"]), s["field"].float(), dim=-1).mean().item()
            rows.append([r.name, key, ours, ref, recompute])
    rep.table(["run", "split", "top1 (dump)", "top1 (results.json)", "field recompute cos"], rows)


def accuracy(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("1. Accuracy decomposition", "`final` is what test.py scores; `backbone` drops the field (frames only); "
                "`field` is the field classifier alone. Train is the 16-shot training split under the eval protocol.")
    rows, out = [], {}
    for r in runs:
        for v in r.variants:
            m = {key: {"top1": top1(r.logits(sp[key], v), r.y(sp[key])), "top5": top5(r.logits(sp[key], v), r.y(sp[key]))}
                 for key in sp}
            m["hm"] = hm(m["base"]["top1"], m["novel"]["top1"])
            out.setdefault(r.name, {})[v] = m
            train = m["train"]["top1"] if "train" in m else "-"
            rows.append([r.name, v, train, m["base"]["top1"], m["novel"]["top1"], m["hm"], m["base"]["top5"], m["novel"]["top5"]])
    rep.table(["run", "variant", "train top1", "base top1", "novel top1", "HM", "base top5", "novel top5"], rows)
    rep.metrics["accuracy"] = out


def tower_swap(rep: Report, runs: list[Run], clip: Run | None, sp: dict) -> None:
    if clip is None:
        return
    rep.section("2. Which tower carries the gain", "Each run's vision embedding (final) scored against its own text, "
                "against zero-shot CLIP's text, and CLIP's vision against the run's text.")
    rows, out = [], {}
    for r in runs:
        if r.is_clip:
            continue
        for vision, text, label in ((r, r, "run / run"), (r, clip, "run / clip"), (clip, r, "clip / run")):
            acc = {k: top1(view_logits(vision.emb(sp[k]), text.classes(sp[k]), 100.0), r.y(sp[k])) for k in ("base", "novel")}
            out.setdefault(r.name, {})[label] = {**acc, "hm": hm(acc["base"], acc["novel"])}
            rows.append([r.name, label, acc["base"], acc["novel"], hm(acc["base"], acc["novel"])])
    rep.table(["run", "vision / text", "base", "novel", "HM"], rows)
    rep.metrics["tower_swap"] = out


def alpha_sweep(rep: Report, runs: list[Run], sp: dict) -> None:
    field_runs = [r for r in runs if r.has_field]
    if not field_runs:
        return
    rep.section("3. Field strength at test time", "frames = f + α·g with α swept at test time only (trained α marked).")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    rows, out = [], {}
    for r in field_runs:
        curve = {"base": [], "novel": []}
        for a in ALPHAS:
            for key in curve:
                s = sp[key]
                curve[key].append(top1(view_logits(r.emb(s, alpha=a), r.classes(s), r.scale), r.y(s)))
        curve["hm"] = [hm(b, n) for b, n in zip(curve["base"], curve["novel"])]
        out[r.name] = {"alpha": ALPHAS, **curve}
        for ax, key in zip(axes, ("base", "novel", "hm")):
            ax.plot(ALPHAS, curve[key], color=r.color, marker="o")
            ax.axvline(r.alpha, color=MUTED, ls=":", lw=1)
        best = max(range(len(ALPHAS)), key=lambda i: curve["hm"][i])
        best_novel = max(range(len(ALPHAS)), key=lambda i: curve["novel"][i])
        trained = hm(*(top1(r.logits(sp[k]), r.y(sp[k])) for k in ("base", "novel")))
        rows.append([r.name, r.alpha, curve["base"][0], curve["novel"][0], curve["hm"][0], trained,
                     f"{curve['hm'][best]:.2f} @ {ALPHAS[best]:g}", f"{curve['novel'][best_novel]:.2f} @ {ALPHAS[best_novel]:g}"])
    for ax, title in zip(axes, ("base top1", "novel top1", "HM")):
        ax.set_title(title)
        ax.set_xlabel("α at test")
    legend_runs(axes[0], field_runs)
    rep.figure(fig, "alpha_sweep", "field strength sweep")
    rep.table(["run", "trained α", "base @0", "novel @0", "HM @0", "HM @trained", "best HM", "best novel"], rows)
    rep.metrics["alpha_sweep"] = out


def zeroshot_ensemble(rep: Report, runs: list[Run], clip: Run | None, sp: dict) -> None:
    if clip is None:
        return
    rep.section("4. Output ensemble with zero-shot CLIP",
                "(1-λ)·logits(run) + λ·logits(CLIP), each in its own text space: how much novel the pretrained model "
                "still adds on top.")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    rows, out, shown = [], {}, [r for r in runs if not r.is_clip]
    for r in shown:
        curve = {"base": [], "novel": []}
        for lam in LAMBDAS:
            for key in curve:
                s = sp[key]
                curve[key].append(top1((1 - lam) * r.logits(s) + lam * clip.logits(s), r.y(s)))
        curve["hm"] = [hm(b, n) for b, n in zip(curve["base"], curve["novel"])]
        out[r.name] = {"lambda": LAMBDAS, **curve}
        for ax, key in zip(axes, ("base", "novel", "hm")):
            ax.plot(LAMBDAS, curve[key], color=r.color, marker="o")
        best = max(range(len(LAMBDAS)), key=lambda i: curve["hm"][i])
        rows.append([r.name, curve["hm"][0], f"{curve['hm'][best]:.2f} @ {LAMBDAS[best]:g}",
                     curve["base"][best], curve["novel"][best], max(curve["novel"])])
    for ax, title in zip(axes, ("base top1", "novel top1", "HM")):
        ax.set_title(title)
        ax.set_xlabel("λ (weight on zero-shot CLIP)")
    legend_runs(axes[0], shown)
    rep.figure(fig, "zeroshot_ensemble", "output ensemble with zero-shot CLIP")
    rep.table(["run", "HM λ=0", "best HM", "base @best", "novel @best", "max novel"], rows)
    rep.metrics["zeroshot_ensemble"] = out


def base_pull(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("5. Base pull in the joint label space",
                "Every video scored against base ∪ novel class texts (26 + 25). `→ other` is the share of a split's "
                "videos whose top-1 lands in the OTHER split's classes; joint HM is the generalized-ZSL HM.")
    rows, out, bars = [], {}, []
    cb = len(runs[0].splits[sp["base"]]["classnames"])
    for r in runs:
        union = torch.cat([r.classes(sp["base"]), r.classes(sp["novel"])])
        for v in r.variants:
            m = {}
            for key, offset in (("base", 0), ("novel", cb)):
                pred = view_logits(r.emb(sp[key], v), union, r.scale).sum(1).argmax(-1)
                other = pred >= cb if key == "base" else pred < cb
                m[key] = {"joint_top1": 100 * (pred == r.y(sp[key]) + offset).float().mean().item(),
                          "to_other": 100 * other.float().mean().item()}
            m["joint_hm"] = hm(m["base"]["joint_top1"], m["novel"]["joint_top1"])
            out.setdefault(r.name, {})[v] = m
            rows.append([r.name, v, m["base"]["joint_top1"], m["base"]["to_other"], m["novel"]["joint_top1"],
                         m["novel"]["to_other"], m["joint_hm"]])
            bars.append((f"{r.name} · {v}", m["novel"]["to_other"], r.color))
    rep.table(["run", "variant", "base joint top1", "base → novel %", "novel joint top1", "novel → base %", "joint HM"], rows)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(bars) + 0.8))
    ys = list(range(len(bars)))[::-1]
    ax.barh(ys, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62, edgecolor=SURFACE, linewidth=2)
    for yy, (_, val, _) in zip(ys, bars):
        ax.text(val + 0.5, yy, f"{val:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, [b[0] for b in bars])
    ax.set_xlabel("novel videos whose joint top-1 is a BASE class (%)")
    ax.grid(axis="y", visible=False)
    rep.figure(fig, "base_pull", "base pull")
    rep.metrics["base_pull"] = out


def field_geometry(rep: Report, runs: list[Run], clip: Run | None, sp: dict) -> None:
    rep.section("6. Field geometry and its effect on the logits",
                "ratio = α‖g‖/‖f‖ per frame; cos(g, f); margin gain = Δ(true logit − mean other logit) from adding "
                "the field (per video, summed views); flips = backbone → final correctness changes.")
    rows, out = [], {}
    field_runs = [r for r in runs if r.has_field]
    for r in field_runs:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for key in [k for k in ("train", "base", "novel") if k in sp]:
            s, y = r.splits[sp[key]], r.y(sp[key])
            f, g = s["head"].float(), s["field"].float()
            ratio = (r.alpha * g.norm(dim=-1) / f.norm(dim=-1)).flatten()
            cos = F.cosine_similarity(g, f, dim=-1).flatten()
            d = (r.logits(sp[key], "final") - r.logits(sp[key], "backbone")).sum(1)
            true = d.gather(1, y[:, None]).squeeze(1)
            gain = (true - (d.sum(1) - true) / (d.shape[1] - 1)).mean().item()
            before = r.logits(sp[key], "backbone").sum(1).argmax(-1) == y
            after = r.logits(sp[key], "final").sum(1).argmax(-1) == y
            m = {"ratio_mean": ratio.mean().item(), "ratio_median": ratio.median().item(), "cos_gf": cos.mean().item(),
                 "margin_gain": gain, "dlogit_std": d.std(1).mean().item(),
                 "fixed": int((~before & after).sum()), "broken": int((before & ~after).sum()), "n": len(y)}
            out.setdefault(r.name, {})[key] = m
            rows.append([r.name, key, m["ratio_mean"], m["ratio_median"], m["cos_gf"], m["margin_gain"], m["dlogit_std"],
                         m["fixed"], m["broken"], m["n"]])
            if key in SPLIT_COLORS:
                axes[0].hist(ratio.numpy(), bins=60, histtype="step", lw=1.6, color=SPLIT_COLORS[key], ls=SPLIT_STYLES[key],
                             density=True, label=key)
                axes[1].hist(cos.numpy(), bins=60, histtype="step", lw=1.6, color=SPLIT_COLORS[key], ls=SPLIT_STYLES[key],
                             density=True, label=key)
        axes[0].set_xlabel("α‖g‖ / ‖f‖ per frame")
        axes[1].set_xlabel("cos(g, f) per frame")
        for ax in axes:
            ax.set_ylabel("density")
            ax.legend()
        fig.suptitle(r.name, fontsize=10, color=INK)
        rep.figure(fig, f"field_geometry_{r.name}", f"field geometry {r.name}")
    if rows:
        rep.table(["run", "split", "ratio mean", "ratio median", "cos(g,f)", "margin gain", "Δlogit std", "fixed", "broken", "n"], rows)

    rep.text("**Text-subspace energy**: % of each unit frame vector's squared norm inside the span of the run's base "
             "class texts (26-d), novel class texts (25-d) and both (51-d); a random direction puts ~5 / 5 / 10 % there. "
             "`Δ vs CLIP` = unit(final) − unit(zero-shot CLIP) on the same frame.")
    rows, energies = [], {}
    for r in runs:
        bases = {"base": text_basis(r.classes(sp["base"])), "novel": text_basis(r.classes(sp["novel"]))}
        bases["both"] = text_basis(torch.cat([r.classes(sp["base"]), r.classes(sp["novel"])]))
        for key in ("base", "novel"):
            quantities = {"embedding": r.emb(sp[key])}
            if clip is not None and not r.is_clip:
                quantities["Δ vs CLIP"] = F.normalize(r.emb(sp[key]), dim=-1) - F.normalize(clip.emb(sp[key]), dim=-1)
            if r.has_field:
                quantities["backbone f"] = r.emb(sp[key], "backbone")
                quantities["field g"] = r.emb(sp[key], "field")
            for q, x in quantities.items():
                e = {b: energy(x, basis) for b, basis in bases.items()}
                energies.setdefault(r.name, {}).setdefault(key, {})[q] = e
                rows.append([r.name, q, key, e["base"], e["novel"], e["both"]])
    rep.table(["run", "quantity", "videos", "in base-text span %", "in novel-text span %", "in both %"], rows)
    rep.metrics["field_geometry"] = out
    rep.metrics["text_energy"] = energies


def field_targets(rep: Report, runs: list[Run], sp: dict) -> None:
    field_runs = [r for r in runs if r.has_field]
    if not field_runs:
        return
    rep.section("7. Where the field points for each novel class",
                "Mean unit field direction over a novel class's frames, matched against all 51 class texts: its nearest "
                "text, and the rank of the true class among the 51.")
    out = {}
    for r in field_runs:
        names = [short(c) for c in r.splits[sp["base"]]["classnames"]] + \
                [short(c) for c in r.splits[sp["novel"]]["classnames"]]
        cb = len(r.splits[sp["base"]]["classnames"])
        union = torch.cat([r.classes(sp["base"]), r.classes(sp["novel"])])
        s, y = r.splits[sp["novel"]], r.y(sp["novel"])
        g = F.normalize(s["field"].float(), dim=-1).mean((1, 2))
        f = F.normalize(s["head"].float(), dim=-1).mean((1, 2))
        final_pred = r.logits(sp["novel"]).sum(1).argmax(-1)
        field_pred = r.logits(sp["novel"], "field").sum(1).argmax(-1)
        rows, to_base = [], 0
        for c in y.unique().tolist():
            m = y == c
            sims = union @ F.normalize(g[m].mean(0), dim=0)
            order = sims.argsort(descending=True)
            nearest = order[0].item()
            rank = (order == cb + c).nonzero().item() + 1
            f_nearest = (union @ F.normalize(f[m].mean(0), dim=0)).argmax().item()
            to_base += nearest < cb
            tag = lambda i: f"{names[i]} ({'base' if i < cb else 'novel'})"  # noqa: E731
            rows.append([names[cb + c], 100 * (final_pred[m] == c).float().mean().item(),
                         100 * (field_pred[m] == c).float().mean().item(), tag(nearest), sims[nearest].item(), rank,
                         tag(f_nearest)])
        rows.sort(key=lambda row: row[2])
        out[r.name] = {"novel_classes_pointing_to_base": to_base, "rows": rows}
        rep.text(f"**{r.name}**: the field's mean direction is nearest a BASE class text for {to_base} of "
                 f"{len(rows)} novel classes.")
        rep.table(["novel class", "final acc", "field acc", "field → nearest text", "cos", "true rank /51",
                   "backbone → nearest text"], rows)
    rep.metrics["field_targets"] = out


def field_scores(r: Run, split: str, pooled: torch.Tensor) -> tuple[float, float]:  # field-alone and final top1
    g, classes, y = field_apply(r.field_embed, pooled), r.classes(split), r.y(split)
    head = r.splits[split]["head"].float()
    return top1(view_logits(g, classes, r.scale), y), top1(view_logits(head + r.alpha * g, classes, r.scale), y)


def field_layers(rep: Report, runs: list[Run], sp: dict) -> None:
    field_runs = [r for r in runs if r.has_field]
    if not field_runs:
        return
    rep.section("8. Which blocks the field reads",
                "Mean-ablation: block ℓ's 64 pooled values replaced by their train-split mean (field alone and final); "
                "only-ℓ: every other block mean-replaced; weight energy = ‖W1[:, ℓ]·std_ℓ‖ share.")
    out = {}
    ref_key = "train" if "train" in sp else "base"
    for r in field_runs:
        ref_pooled = r.splits[sp[ref_key]]["pooled"].float()
        depth = ref_pooled.shape[-2]
        ref = ref_pooled.reshape(-1, depth, ref_pooled.shape[-1]).mean(0)
        std = ref_pooled.reshape(-1, depth, ref_pooled.shape[-1]).std(0)
        w1 = r.field_embed["net.1.weight"].view(-1, depth, ref.shape[-1])
        weight = torch.stack([(w1[:, l] * std[l]).norm() for l in range(depth)])
        weight = (100 * weight / weight.sum()).tolist()
        res = {"weight_energy": weight}
        for key in ("base", "novel"):
            pooled = r.splits[sp[key]]["pooled"].float()
            full = field_scores(r, sp[key], pooled)
            ablate, only = [], []
            for l in range(depth):
                p = pooled.clone()
                p[..., l, :] = ref[l]
                ablate.append(field_scores(r, sp[key], p))
                p = ref.expand_as(pooled).clone()
                p[..., l, :] = pooled[..., l, :]
                only.append(field_scores(r, sp[key], p)[0])
            res[key] = {"full_field": full[0], "full_final": full[1], "ablate_field": [a[0] for a in ablate],
                        "ablate_final": [a[1] for a in ablate], "only_field": only}
        out[r.name] = res
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        xs = list(range(1, depth + 1))
        for key in ("base", "novel"):
            kw = dict(color=SPLIT_COLORS[key], ls=SPLIT_STYLES[key], marker="o", label=key)
            axes[0, 0].plot(xs, [a - res[key]["full_field"] for a in res[key]["ablate_field"]], **kw)
            axes[0, 1].plot(xs, [a - res[key]["full_final"] for a in res[key]["ablate_final"]], **kw)
            axes[1, 0].plot(xs, res[key]["only_field"], **kw)
        axes[1, 1].bar(xs, weight, color=INK2, width=0.7, edgecolor=SURFACE, linewidth=2)
        for ax, title in zip(axes.flat, ("field alone: Δtop1 when block ℓ is mean-ablated",
                                         "final: Δtop1 when block ℓ is mean-ablated",
                                         "field alone: top1 with only block ℓ kept", "weight energy share (%)")):
            ax.set_title(title)
            ax.set_xlabel("block ℓ")
            ax.set_xticks(xs)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.legend()
        axes[1, 1].grid(axis="x", visible=False)
        fig.suptitle(r.name, fontsize=10, color=INK)
        fig.tight_layout()
        rep.figure(fig, f"field_layers_{r.name}", f"field layer attribution {r.name}")
        rep.table(["block"] + [str(x) for x in xs],
                  [["weight %"] + weight] +
                  [[f"{k} ablate Δfield"] + [a - res[k]["full_field"] for a in res[k]["ablate_field"]] for k in ("base", "novel")] +
                  [[f"{k} ablate Δfinal"] + [a - res[k]["full_final"] for a in res[k]["ablate_final"]] for k in ("base", "novel")] +
                  [[f"{k} only-ℓ field"] + res[k]["only_field"] for k in ("base", "novel")])

        rep.text(f"**Transplant**: {r.name}'s field_embed applied to other models' pooled block CLS (scored with each "
                 f"model's own text; final = that model's f + {r.alpha:g}·g).")
        rows = []
        for other in runs:
            if other.segments != r.segments:
                continue
            acc = {}
            for key in ("base", "novel"):
                s = other.splits[sp[key]]
                g = field_apply(r.field_embed, s["pooled"])
                acc[key] = (top1(view_logits(g, other.classes(sp[key]), r.scale), other.y(sp[key])),
                            top1(view_logits(s["head"].float() + r.alpha * g, other.classes(sp[key]), r.scale), other.y(sp[key])))
            rows.append([other.name, acc["base"][0], acc["novel"][0], acc["base"][1], acc["novel"][1],
                         hm(acc["base"][1], acc["novel"][1])])
            res.setdefault("transplant", {})[other.name] = acc
        rep.table(["pooled from", "field base", "field novel", "final base", "final novel", "final HM"], rows)
    rep.metrics["field_layers"] = out


def separability(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("9. Class separability per block (nearest class mean, 5-fold within each split)",
                "Label-supervised upper bound on what a readout of each block could separate, independent of the text "
                "tower. Novel NCM far above novel zero-shot top-1 means the information is there but not aligned to text.")
    out, rows = {}, []
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for r in runs:
        res = {}
        for j, key in enumerate(("base", "novel")):
            s, y = r.splits[sp[key]], r.y(sp[key])
            depth = s["cls"].shape[1]
            curve = {tok: [ncm(s[tok][:, l], y) for l in range(depth)] for tok in ("cls", "patch")}
            ends = {"embedding": ncm(video_mean(r.emb(sp[key])), y),
                    "pooled": ncm(s["pooled"].float().mean((1, 2)).flatten(1), y)}
            if r.has_field:
                ends["backbone f"] = ncm(video_mean(r.emb(sp[key], "backbone")), y)
                ends["field g"] = ncm(video_mean(r.emb(sp[key], "field")), y)
            res[key] = {**curve, **ends}
            for i, tok in enumerate(("cls", "patch")):
                axes[i, j].plot(range(1, depth + 1), curve[tok], color=r.color, marker="o")
            rows.append([r.name, key, curve["cls"][-1], curve["patch"][-1],
                         f"{max(curve['cls']):.2f} @ {curve['cls'].index(max(curve['cls'])) + 1}",
                         ends["pooled"], ends["embedding"], ends.get("backbone f", "-"), ends.get("field g", "-")])
        out[r.name] = res
    for i, tok in enumerate(("CLS", "patch mean")):
        for j, key in enumerate(("base", "novel")):
            axes[i, j].set_title(f"{tok} · {key}")
            axes[i, j].set_ylabel("NCM top1")
    for ax in axes[1]:
        ax.set_xlabel("block")
        ax.set_xticks(range(1, 13))
    legend_runs(axes[0, 0], runs)
    fig.tight_layout()
    rep.figure(fig, "layer_ncm", "class separability per block")
    rep.table(["run", "split", "CLS @12", "patch @12", "best CLS @block", "field input (pooled)", "embedding",
               "backbone f", "field g"], rows)
    rep.metrics["separability"] = out


def drift(rep: Report, runs: list[Run], clip: Run | None, sp: dict) -> None:
    rep.section("10. Drift from pretrained CLIP per block, and the KI stream",
                "Linear CKA between a run's per-video block tokens and zero-shot CLIP's on the same videos; "
                "for KI runs, the student stream vs the frozen stream at each injection point (CLS, video mean).")
    out = {}
    if clip is not None:
        shown = [r for r in runs if not r.is_clip]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
        for r in shown:
            for key in ("base", "novel"):
                a, b = r.splits[sp[key]], clip.splits[sp[key]]
                depth = a["cls"].shape[1]
                curve = {tok: [cka(a[tok][:, l], b[tok][:, l]) for l in range(depth)] for tok in ("cls", "patch")}
                out.setdefault(r.name, {})[key] = curve
                for ax, tok in zip(axes, ("cls", "patch")):
                    ax.plot(range(1, depth + 1), curve[tok], color=r.color, ls=SPLIT_STYLES[key], marker="o")
        for ax, tok in zip(axes, ("CLS", "patch mean")):
            ax.set_title(f"CKA to zero-shot CLIP · {tok}")
            ax.set_xlabel("block")
            ax.set_xticks(range(1, 13))
        legend_runs(axes[0], shown, styles=True)
        rep.figure(fig, "layer_drift", "drift from CLIP per block")
        rep.table(["run", "split"] + [f"CLS b{l}" for l in (1, 4, 8, 10, 11, 12)] + [f"patch b{l}" for l in (4, 8, 12)],
                  [[name, key] + [m[key]["cls"][l - 1] for l in (1, 4, 8, 10, 11, 12)] + [m[key]["patch"][l - 1] for l in (4, 8, 12)]
                   for name, m in out.items() for key in ("base", "novel")])
    ki = [r for r in runs if "stream" in r.splits[sp["base"]]]
    if ki:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        stream_out = {}
        for r in ki:
            for key in ("base", "novel"):
                s = r.splits[sp[key]]
                h, t = s["cls"][:, :-1], s["stream"][:, 1:]  # block i+1 averages the student's block-i output with t_i
                xs = list(range(2, h.shape[1] + 2))
                curve = {"cos": F.cosine_similarity(h, t, dim=-1).mean(0).tolist(),
                         "cka": [cka(h[:, l], t[:, l]) for l in range(h.shape[1])],
                         "norm_ratio": (h.norm(dim=-1) / t.norm(dim=-1)).mean(0).tolist()}
                stream_out.setdefault(r.name, {})[key] = curve
                for ax, q in zip(axes, ("cos", "cka", "norm_ratio")):
                    ax.plot(xs, curve[q], color=r.color, ls=SPLIT_STYLES[key], marker="o")
        for ax, title in zip(axes, ("cos(student, frozen)", "CKA(student, frozen)", "‖student‖ / ‖frozen‖")):
            ax.set_title(title)
            ax.set_xlabel("injection before block")
            ax.set_xticks(range(2, 13))
        legend_runs(axes[0], ki, styles=True)
        rep.figure(fig, "ki_stream", "student vs frozen stream at each injection point")
        rep.metrics["ki_stream"] = stream_out
    rep.metrics["drift"] = out


def text_drift(rep: Report, runs: list[Run], clip: Run | None, sp: dict) -> None:
    rep.section("11. Text tower", "cos(run class text, CLIP class text) per class; crowding = mean pairwise cosine "
                "between a split's class texts; nearest = mean cosine to the nearest other class in the same split.")
    rows, out, labelled = [], {}, set()
    shown = [r for r in runs if not r.is_clip]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for r in runs:
        for key in ("base", "novel"):
            c = r.classes(sp[key])
            sim = c @ c.T
            off = sim[~torch.eye(len(c), dtype=torch.bool)]
            nearest = sim.fill_diagonal_(-1).max(1).values.mean().item()
            m = {"crowding": off.mean().item(), "nearest": nearest}
            if clip is not None and not r.is_clip:
                per = (c * clip.classes(sp[key])).sum(-1)
                m.update({"to_clip_mean": per.mean().item(), "to_clip_min": per.min().item()})
                jitter = torch.linspace(-0.12, 0.12, len(per)) + shown.index(r) + (0.18 if key == "novel" else -0.18)
                ax.scatter(jitter, per, s=14, color=SPLIT_COLORS[key], label=None if key in labelled else key,
                           edgecolor=SURFACE, linewidth=0.6)
                labelled.add(key)
            out.setdefault(r.name, {})[key] = m
            rows.append([r.name, key, m.get("to_clip_mean", "-"), m.get("to_clip_min", "-"), m["crowding"], m["nearest"]])
    ax.set_xticks(range(len(shown)), [r.name for r in shown])
    ax.set_ylabel("cos to CLIP class text")
    if labelled:
        ax.legend()
    rep.figure(fig, "text_drift", "text drift per class")
    rep.table(["run", "split", "cos to CLIP mean", "cos to CLIP min", "crowding", "nearest"], rows)
    rep.metrics["text"] = out


def confidence(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("12. Confidence and calibration", "Softmax of the view-averaged logits (final). ECE with 15 bins; "
                "margin = top1 − top2 probability.")
    rows, out = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for r in runs:
        for ax, key in zip(axes, ("base", "novel")):
            m = calibration(r.logits(sp[key]), r.y(sp[key]))
            out.setdefault(r.name, {})[key] = m
            rows.append([r.name, key, m["acc"], m["conf"], m["ece"], m["margin"], m["conf_right"], m["conf_wrong"]])
            ax.plot([c[0] * 100 for c in m["curve"]], [c[1] for c in m["curve"]], color=r.color, marker="o")
    for ax, key in zip(axes, ("base", "novel")):
        ax.plot([0, 100], [0, 100], color=AXIS, lw=1)
        ax.set_title(f"reliability · {key}")
        ax.set_xlabel("confidence (%)")
        ax.set_ylabel("accuracy (%)")
    legend_runs(axes[0], runs)
    rep.figure(fig, "reliability", "reliability")
    rep.table(["run", "split", "acc", "mean conf", "ECE", "margin", "conf if right", "conf if wrong"], rows)
    rep.metrics["confidence"] = out


def structure(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("13. Embedding structure", "Per-video unit embeddings: effective rank (exp entropy of the centred "
                "spectrum) and Fisher ratio (between- over within-class scatter, higher = tighter classes).")
    rows, out = [], {}
    for r in runs:
        for v in r.variants:
            for key in ("base", "novel"):
                x = video_mean(r.emb(sp[key], v))
                m = {"erank": erank(x), "fisher": fisher(x, r.y(sp[key]))}
                out.setdefault(r.name, {}).setdefault(v, {})[key] = m
                rows.append([r.name, v, key, m["erank"], m["fisher"]])
    rep.table(["run", "variant", "split", "effective rank", "Fisher ratio"], rows)
    rep.metrics["structure"] = out


def per_class(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("14. Per-class accuracy", "Final top-1 per class; CSVs next to this report. Classes sorted by the last "
                "run's accuracy.")
    out = {}
    for key in ("base", "novel"):
        names = [short(c) for c in runs[0].splits[sp[key]]["classnames"]]
        y = runs[0].y(sp[key])
        cols = {}
        for r in runs:
            for v in r.variants:
                pred = r.logits(sp[key], v).sum(1).argmax(-1)
                cols[r.name if v == "final" else f"{r.name}:{v}"] = [
                    100 * (pred[y == c] == c).float().mean().item() for c in range(len(names))]
        out[key] = cols
        with open(rep.out / f"per_class_{key}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "n"] + list(cols))
            for c, name in enumerate(names):
                w.writerow([name, int((y == c).sum())] + [f"{cols[k][c]:.2f}" for k in cols])
        last = cols[runs[-1].name]
        order = sorted(range(len(names)), key=lambda c: last[c])
        fig, ax = plt.subplots(figsize=(7, 0.24 * len(names) + 1))
        for k, r in enumerate(runs):
            ys = [i + (k - (len(runs) - 1) / 2) * 0.16 for i in range(len(names))]
            ax.scatter([cols[r.name][c] for c in order], ys, s=22, color=r.color, label=r.name, edgecolor=SURFACE,
                       linewidth=0.6, zorder=3)
        ax.set_yticks(range(len(names)), [names[c] for c in order])
        ax.set_xlabel(f"{key} top1 per class (%)")
        ax.set_xlim(-3, 103)
        ax.legend(loc="lower right")
        rep.figure(fig, f"per_class_{key}", f"per-class {key}")
        if len(runs) >= 2:
            prev, now = runs[-2].name, runs[-1].name
            delta = sorted(range(len(names)), key=lambda c: cols[now][c] - cols[prev][c])
            picks = delta[:5] + delta[-5:][::-1]
            rep.text(f"**{key}: largest changes {prev} → {now}** (losses first, then gains)")
            rep.table(["class"] + list(cols) + ["Δ"],
                      [[names[c]] + [cols[k][c] for k in cols] + [cols[now][c] - cols[prev][c]] for c in picks])
    rep.metrics["per_class"] = out


def transitions(rep: Report, runs: list[Run], sp: dict) -> None:
    rep.section("15. Correctness transitions", "Videos (summed over views) right in both, gained, lost, wrong in both, "
                "for consecutive runs and inside each field run.")
    pairs = [((a, "final"), (b, "final")) for a, b in zip(runs, runs[1:])]
    for r in runs:
        if r.has_field:
            pairs += [((r, "backbone"), (r, "final")), ((r, "backbone"), (r, "field"))]
    rows = []
    for (a, va), (b, vb) in pairs:
        for key in ("base", "novel"):
            y = a.y(sp[key])
            ra = a.logits(sp[key], va).sum(1).argmax(-1) == y
            rb = b.logits(sp[key], vb).sum(1).argmax(-1) == y
            rows.append([f"{a.name}:{va} → {b.name}:{vb}", key, int((ra & rb).sum()), int((~ra & rb).sum()),
                         int((ra & ~rb).sum()), int((~ra & ~rb).sum())])
    rep.table(["pair", "split", "both right", "gained", "lost", "both wrong"], rows)


def confusions(rep: Report, runs: list[Run], sp: dict, top: int = 8) -> None:
    rep.section("16. Top novel confusions", "true → predicted (final), with the cosine between the two class texts "
                "in that run's text space.")
    out = {}
    for r in runs:
        s, y = r.splits[sp["novel"]], r.y(sp["novel"])
        names = [short(c) for c in s["classnames"]]
        pred = r.logits(sp["novel"]).sum(1).argmax(-1)
        c = r.classes(sp["novel"])
        pairs: dict[tuple[int, int], int] = {}
        for t, p in zip(y.tolist(), pred.tolist()):
            if t != p:
                pairs[(t, p)] = pairs.get((t, p), 0) + 1
        ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:top]
        rows = [[names[t], names[p], n, 100 * n / int((y == t).sum()), (c[t] @ c[p]).item()] for (t, p), n in ranked]
        out[r.name] = rows
        rep.text(f"**{r.name}**")
        rep.table(["true", "predicted", "count", "% of class", "text cos"], rows)
    rep.metrics["confusions"] = out


def curves(rep: Report, runs: list[Run]) -> None:
    histories = {}
    for r in runs:
        path = r.run_dir() / "history.json" if r.run_dir() is not None else None
        if path is not None and path.exists():
            histories[r.name] = (r, json.loads(path.read_text()))
    if not histories:
        return
    rep.section("17. Training curves", "Per-epoch means from history.json.")
    keys = [("ce", "CE (main)"), ("kd", "distillation"), ("field_ce", "field CE"), ("top1", "train top1")]
    fig, axes = plt.subplots(1, len(keys), figsize=(13, 3))
    for ax, (k, title) in zip(axes, keys):
        for r, hist in histories.values():
            if all(k in e for e in hist):
                ax.plot([e["epoch"] for e in hist], [e[k] for e in hist], color=r.color, marker="o")
        ax.set_title(title)
        ax.set_xlabel("epoch")
    legend_runs(axes[0], [r for r, _ in histories.values()])
    fig.tight_layout()
    rep.figure(fig, "training_curves", "training curves")


def main() -> None:
    ap = argparse.ArgumentParser(description="analysis report over scripts/analysis/dump.py outputs")
    ap.add_argument("--dumps", default="runs/analysis/dumps")
    ap.add_argument("--runs", nargs="+", default=None, help="dump names in order (default: every dump, clip first)")
    ap.add_argument("--train", default="b2n/hmdb51/train_s1")
    ap.add_argument("--base", default="b2n/hmdb51/base")
    ap.add_argument("--novel", default="b2n/hmdb51/novel")
    ap.add_argument("--out", default="docs/analysis")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dumps = Path(args.dumps)
    names = args.runs or sorted((p.stem for p in dumps.glob("*.pt")), key=lambda n: (n != CLIP_NAME, n))
    if CLIP_NAME not in names and (dumps / f"{CLIP_NAME}.pt").exists():
        names = [CLIP_NAME] + names
    runs = [Run(dumps / f"{n}.pt") for n in names]
    clip = next((r for r in runs if r.is_clip), None)
    for i, r in enumerate([r for r in runs if not r.is_clip]):
        r.color = SERIES[i % len(SERIES)]
    sp = {"train": args.train, "base": args.base, "novel": args.novel}
    sp = {k: v for k, v in sp.items() if all(v in r.splits for r in runs)}
    if "base" not in sp or "novel" not in sp:
        raise SystemExit(f"every dump needs {args.base} and {args.novel}; rerun dump.py with those splits")
    for key, split in sp.items():
        ref = runs[0].y(split)
        if any(not torch.equal(r.y(split), ref) for r in runs):
            raise SystemExit(f"{split}: dumps disagree on video order; dump every model in one dump.py call")

    style()
    rep = Report(Path(args.out))
    rep.lines += [f"# Analysis: {', '.join(r.name for r in runs)}", "",
                  f"Dumps: `{dumps}` | splits: " + ", ".join(f"{k} `{v}`" for k, v in sp.items())]
    for step in (sanity, accuracy):
        step(rep, runs, sp)
    tower_swap(rep, runs, clip, sp)
    alpha_sweep(rep, runs, sp)
    zeroshot_ensemble(rep, runs, clip, sp)
    base_pull(rep, runs, sp)
    field_geometry(rep, runs, clip, sp)
    for step in (field_targets, field_layers, separability):
        step(rep, runs, sp)
    drift(rep, runs, clip, sp)
    text_drift(rep, runs, clip, sp)
    for step in (confidence, structure, per_class, transitions, confusions):
        step(rep, runs, sp)
    curves(rep, runs)
    rep.write()
    print(f"  wrote {rep.out}/report.md, metrics.json, figures/ and per_class_*.csv")


if __name__ == "__main__":
    main()
