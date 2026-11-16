#!/usr/bin/env python3
# Regenerates every figure in docs/cgas.md from runs/, the archived pooling/capacity runs under
# ARCHIVE_RUNS, and their dumps under ARCHIVE_DUMPS. Computed data is cached in runs/analysis/cgas;
# pass --recompute after new runs land.
#
#     python scripts/analysis/cgas_figures.py [--recompute]

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from dataclasses import fields, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import ModelConfig, load_config  # noqa: E402
from src.models.cgas import pool_segments  # noqa: E402
from src.models.kiclip import build_kiclip  # noqa: E402

RUNS = REPO_ROOT / "runs"
ARCHIVE = Path("~/scratch/KICLIP").expanduser()
ARCHIVE_RUNS = ARCHIVE / "runs_archive"
ARCHIVE_DUMPS = ARCHIVE / "analysis_archive" / "dumps"
CACHE = RUNS / "analysis" / "cgas"
OUT = REPO_ROOT / "docs" / "figures"
TRAIN, BASE, NOVEL = "b2n/hmdb51/train_s1", "b2n/hmdb51/base", "b2n/hmdb51/novel"
# (run dir, pooled width); runs live under ARCHIVE_RUNS except where a plain run() in runs/ is used
POOL_RUNS = {"mean": ("field_pool_mean_last1", 64), "dropout": ("field_layers_last1", 64),
             "max": ("field_pool_max_last1", 64), "min": ("field_pool_min_last1", 64), "none": ("field_nopool", 768)}
MASKS = 64
WIDTH_RUNS = {16: "field_mean_seg16_last1", 32: "field_mean_seg32_last1", 64: "field_pool_mean_last1",
              128: "field_mean_seg128_last1", 256: "field_mean_seg256_last1", 768: "field_nopool"}
SEED_RUNS = {"dd": ["dd", "dd_s1", "dd_s2"],
             **{m: [n, f"{n}_s1", f"{n}_s2"] for m, (n, _) in POOL_RUNS.items()}}
HEAD = ["model.visual.ln_post.weight", "model.visual.ln_post.bias", "model.visual.proj", "model.logit_scale"]
CGAS_HEAD = ["cgas_head.net.1.weight", "cgas_head.net.1.bias", "cgas_head.net.3.weight", "cgas_head.net.3.bias"]

BLUE, ORANGE, AQUA, VIOLET, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#d55181", "#898781"
INK, SUB, PANEL = "#1f1f1f", "#5f5e5a", "#f2f4f8"
MODE_STYLE = {"mean": ("Mean", BLUE), "dropout": ("Dropout", MAGENTA), "max": ("Max", ORANGE), "min": ("Min", AQUA),
              "none": ("No Pooling", VIOLET)}
HALO = [pe.Stroke(linewidth=3.8, foreground="white"), pe.Normal()]
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#0d366b", "#2a78d6", "#9ec5f4", "#f5f5f3", "#f2aaa9", "#e34948", "#8f2423"])
W1, H1 = 5.2, 3.3


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.facecolor": PANEL, "figure.facecolor": "white", "axes.linewidth": 0,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "white", "grid.linewidth": 1.0,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.pad": 4, "ytick.major.pad": 4, "xtick.color": SUB, "ytick.color": SUB,
        "axes.labelcolor": INK, "axes.labelpad": 5, "text.color": INK,
        "legend.frameon": True, "legend.fancybox": True, "legend.framealpha": 0.95, "legend.facecolor": "white",
        "legend.edgecolor": "#dde2ea", "legend.borderpad": 0.6, "legend.handlelength": 1.8,
        "lines.linewidth": 1.7, "lines.solid_capstyle": "round",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "savefig.facecolor": "white",
    })


def tint(color: str, amount: float) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount


def legend(ax, **kw) -> None:
    ax.legend(**kw).get_frame().set_linewidth(0.6)


def marker_kw(c: str) -> dict:
    return dict(marker="o", markersize=4.6, markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.9)


def series(ax, x, y, c: str, label: str | None = None) -> Line2D:
    ax.plot(x, y, color=c, zorder=4, path_effects=HALO)
    ax.plot(x, y, linestyle="", zorder=4.5, **marker_kw(c))
    return Line2D([], [], color=c, label=label, **marker_kw(c))


def gradient_fill(ax, x, y, color: str, base: float, top_alpha: float = 0.22) -> None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    grad = np.zeros((256, 1, 4))
    grad[..., :3] = to_rgb(color)
    grad[..., 3] = np.linspace(top_alpha, 0, 256)[:, None]
    im = ax.imshow(grad, aspect="auto", extent=[x.min(), x.max(), base, y.max()], origin="upper", zorder=1)
    im.set_clip_path(ax.fill_between(x, y, base, facecolor="none", edgecolor="none").get_paths()[0],
                     transform=ax.transData)


def run_dir(name: str) -> Path:
    # most pooling/capacity runs were archived to scratch when field_alpha's test-time path was removed;
    # anything still under runs/ (dd, froster, cgas itself) is checked first
    local = RUNS / name
    return local if local.exists() else ARCHIVE_RUNS / name


def dump_path(name: str) -> Path:
    local = RUNS / "analysis" / "dumps" / f"{name}.pt"
    return local if local.exists() else ARCHIVE_DUMPS / f"{name}.pt"


def hm(name: str) -> float:
    return json.load(open(run_dir(name) / "results.json"))["HM"]


def cached(name: str, fn, recompute: bool) -> dict:
    path = CACHE / f"{name}.npz"
    if path.exists() and not recompute:
        return dict(np.load(path))
    data = {k: np.asarray(v) for k, v in fn().items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def load_weights(name: str, keys: list[str]) -> dict[str, torch.Tensor]:
    ckpt = torch.load(run_dir(name) / "averaged.pt", map_location="cpu", mmap=True, weights_only=False)
    out = {k: ckpt["model"][k].clone().float() for k in keys}
    del ckpt
    gc.collect()
    return out


def cgas_logits(w: dict, cls: torch.Tensor, seg: int, mode: str, w1: torch.Tensor) -> torch.Tensor:
    hidden = F.gelu(F.linear(pool_segments(cls, seg, mode), w1, w["cgas_head.net.1.bias"]))
    return F.normalize(F.linear(hidden, w["cgas_head.net.3.weight"], w["cgas_head.net.3.bias"]), dim=-1)


def impulse_data() -> dict:
    # one real frame from each of 48 HMDB classes, a norm-5 kick to block 1's CLS input, tracked through all 12 blocks
    cfg = load_config([str(REPO_ROOT / "configs/b2n/default/hmdb51.yaml")])
    seen, videos = set(), []
    for split in (BASE, NOVEL):
        for line in cfg.split_csv(split).read_text().splitlines():
            rel = line.split(",")[0]
            if rel.split("/")[0] not in seen and len(videos) < 48:
                seen.add(rel.split("/")[0])
                videos.append(cfg.video_root(split) / rel)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, v in enumerate(videos):
            png = Path(tmp) / f"{i}.png"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "1", "-i", str(v), "-frames:v", "1",
                            "-vf", "scale=-2:224,crop=224:224", "-y", str(png)], check=True)
            rgb = torch.from_numpy(np.array(Image.open(png).convert("RGB"))).permute(2, 0, 1).float() / 255
            frames.append((rgb - mean) / std)
    imgs = torch.stack(frames)

    ckpt = torch.load(run_dir("dd") / "averaged.pt", map_location="cpu", weights_only=False)
    known = {f.name for f in fields(ModelConfig)}
    cfg = replace(cfg, model=ModelConfig(**{k: v for k, v in ckpt["config"]["model"].items() if k in known}))
    model = build_kiclip(cfg, {k[len("model."):]: v for k, v in ckpt["model"].items() if k.startswith("model.")})
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    del ckpt
    visual = model.model.visual
    blocks, frozen = visual.transformer.resblocks, visual.transformer.resblocks_dd

    def run(embed, kick, damp):
        x, t = embed.clone(), embed.clone()
        x[:, 0] += kick
        out = []
        for i, blk in enumerate(blocks):
            if damp:
                x, t = (x + t) / 2, frozen[i](t)
            x = blk(x)
            out.append(x[:, 0])
        return torch.stack(out, 1)

    gen = torch.Generator().manual_seed(0)
    on, off = [], []
    with torch.no_grad():
        for chunk in imgs.split(16):
            embed = visual.embed(chunk)
            kick = torch.randn(embed.shape[0], embed.shape[-1], generator=gen)
            kick = 5.0 * kick / kick.norm(dim=-1, keepdim=True)
            zero = torch.zeros_like(kick)
            on.append((run(embed, kick, True) - run(embed, zero, True)).norm(dim=-1))
            off.append((run(embed, kick, False) - run(embed, zero, False)).norm(dim=-1))
    del model
    gc.collect()
    return {"with_dd": torch.cat(on).numpy(), "without_dd": torch.cat(off).numpy()}


def contraction_data() -> dict:
    s = torch.load(dump_path("dd"), map_location="cpu", mmap=True, weights_only=True)["splits"][TRAIN]
    cls, stream = s["cls"].float(), s["stream"].float()
    return {"before": torch.stack([(cls[:, l - 1] - stream[:, l]).norm(dim=-1) for l in range(1, 12)], 1).numpy()}


def pooling_data() -> dict:
    # every statistic uses each run's own weights on its own dumped features
    res = {}
    for mode, (name, seg) in POOL_RUNS.items():
        w = load_weights(name, HEAD + CGAS_HEAD)
        scale = w["model.logit_scale"].exp()
        dump = torch.load(dump_path(name), map_location="cpu", mmap=True, weights_only=True)
        s = dump["splits"][TRAIN]
        cls, classes, labels = s["cls"][:, 11].float(), s["classes"].float(), s["labels"]

        h = cls.clone().requires_grad_(True)
        z = F.layer_norm(h, (768,), w["model.visual.ln_post.weight"], w["model.visual.ln_post.bias"]) @ w["model.visual.proj"]
        F.cross_entropy(scale * F.normalize(z, dim=-1) @ classes.T, labels).backward()
        g_ce = h.grad.clone()

        def step(pool, seed):
            # one training-time backward pass of cgas_ce; dropout draws a fresh mask per seed
            torch.manual_seed(seed)
            h = cls.clone().requires_grad_(True)
            w1 = w["cgas_head.net.1.weight"].clone().requires_grad_(True)
            logits = scale * cgas_logits(w, h, seg, pool, w1) @ classes.T
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            acc = 100 * (logits.argmax(-1) == labels).float().mean().item()
            return h.grad.clone(), h.grad.norm().item() / w1.grad.norm().item(), loss.item(), acc

        pool = "mean" if mode == "none" else mode
        steps = [step(pool, k) for k in range(MASKS if mode == "dropout" else 1)]
        g_f = steps[0][0]
        a = g_f.abs().sort(dim=-1, descending=True).values
        res[f"{mode}_cos"] = F.cosine_similarity(g_ce, g_f, dim=-1).numpy()
        res[f"{mode}_cum"] = (a.cumsum(-1) / a.sum(-1, keepdim=True).clamp(min=1e-20)).mean(0).numpy()
        res[f"{mode}_g0"] = g_f[0].numpy()
        res[f"{mode}_fraczero"] = (g_f == 0).float().mean().item()
        res[f"{mode}_top10"] = (a[:, :77].sum(-1) / a.sum(-1)).mean().item()
        res[f"{mode}_ratio"] = np.mean([s[1] for s in steps])
        res[f"{mode}_train_loss"] = np.mean([s[2] for s in steps])
        res[f"{mode}_train_acc"] = np.mean([s[3] for s in steps])
        res[f"{mode}_grad_norm"] = g_f.norm(dim=-1).mean().item()
        if mode == "dropout":
            grads = torch.stack([s[0] for s in steps])
            res["dropout_masks"] = np.array([1, 2, 4, 8, 16, 32, 64])
            res["dropout_masks_cos"] = np.array([F.cosine_similarity(g_ce, grads[:n].mean(0), dim=-1).mean().item()
                                                 for n in res["dropout_masks"]])
            res["dropout_vs_mean_cos"] = F.cosine_similarity(grads.mean(0), step("mean", 0)[0], dim=-1).mean().item()
            res["dropout_mask_to_mask_cos"] = F.cosine_similarity(grads[0], grads[1], dim=-1).mean().item()
            _, _, res["dropout_eval_loss"], res["dropout_eval_acc"] = step("mean", 0)
        for split in (BASE, NOVEL):
            t = dump["splits"][split]
            head, cl, lb = t["head"].float(), t["classes"].float(), t["labels"]
            logits = (scale * F.normalize(head, dim=-1) @ cl.T).mean(2).sum(1)
            res[f"{mode}_{split.split('/')[-1]}_final"] = 100 * (logits.argmax(-1) == lb).float().mean().item()
        del dump
        gc.collect()
    return res


def capacity_data() -> dict:
    s = torch.load(dump_path("field_pool_mean_last1"), map_location="cpu", mmap=True, weights_only=True)["splits"]
    tr, nv = s[TRAIN], s[NOVEL]
    out = {"width": [], "ratio": [], "cgas_novel": [], "train_acc": [], "hm": []}
    for seg, name in WIDTH_RUNS.items():
        w = load_weights(name, ["model.logit_scale"] + CGAS_HEAD)
        scale = w["model.logit_scale"].exp()
        h = tr["cls"][:, 11].float().requires_grad_(True)
        w1 = w["cgas_head.net.1.weight"].clone().requires_grad_(True)
        logits = scale * cgas_logits(w, h, seg, "mean", w1) @ tr["classes"].float().T
        F.cross_entropy(logits, tr["labels"]).backward()
        with torch.no_grad():
            pred = (cgas_logits(w, nv["cls"][:, 11].float(), seg, "mean", w1) @ nv["classes"].float().T).argmax(-1)
        for k, v in zip(out, (seg, h.grad.norm().item() / w1.grad.norm().item(),
                              100 * (pred == nv["labels"]).float().mean().item(),
                              100 * (logits.argmax(-1) == tr["labels"]).float().mean().item(), hm(name))):
            out[k].append(v)
    return out


def fig_fingerprint(pool: dict) -> None:
    groups = 32
    fig, axes = plt.subplots(1, 5, figsize=(7.8, 3.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, mode in zip(axes, ("mean", "dropout", "max", "min", "none")):
        g = pool[f"{mode}_g0"][: groups * 12].reshape(groups, 12)
        mesh = ax.pcolormesh(np.arange(13) + 0.5, np.arange(groups + 1) + 0.5, g / np.abs(g).max(), cmap=DIVERGING,
                             vmin=-1, vmax=1, edgecolors="white", linewidth=0.35, rasterized=True)
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(groups + 0.5, 0.5)
        ax.set_xticks([1, 6, 12])
        ax.set_title(MODE_STYLE[mode][0], fontweight="bold", pad=6)
    axes[0].set_yticks([1, 8, 16, 24, 32])
    axes[0].set_ylabel("Pooling Group")
    fig.supxlabel("Position in Group", fontsize=8.5, y=0.035)
    cax = fig.add_axes([0.932, 0.2, 0.01, 0.6])
    cb = fig.colorbar(mesh, cax=cax, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=SUB, labelsize=7.5)
    cb.set_label("Normalized Gradient", fontsize=8.5, labelpad=4)
    fig.subplots_adjust(left=0.065, right=0.915, bottom=0.12, top=0.9)
    fig.savefig(OUT / "gradient_fingerprint.png")
    plt.close(fig)


def fig_impulse(imp: dict) -> None:
    L = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(W1, H1))
    handles = []
    for key, c, lab in (("without_dd", ORANGE, "Without Drift Damping"), ("with_dd", BLUE, "With Drift Damping")):
        v = imp[key]
        ax.fill_between(L, np.percentile(v, 10, 0), np.percentile(v, 90, 0), color=tint(c, 0.55), alpha=0.55,
                        linewidth=0, zorder=2)
        handles.append(series(ax, L, np.median(v, 0), c, lab))
    ideal = np.median(imp["with_dd"], 0)[0] * 0.5 ** (L - 1)
    handles += ax.plot(L, ideal, color=GRAY, linewidth=1.1, linestyle=(0, (3.5, 2.5)), zorder=3,
                       label="Ideal ×½ per block")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(5e-4, 2e1)
    ax.set_xticks(L)
    ax.grid(which="minor", visible=False)
    ax.set_xlabel("Block")
    ax.set_ylabel("Perturbation Norm")
    legend(ax, handles=handles, loc="lower left")
    fig.savefig(OUT / "dd_impulse_response.png")
    plt.close(fig)


def fig_contraction(con: dict) -> None:
    before = con["before"]
    after = before / 2
    B = np.arange(2, 13)
    fig, ax = plt.subplots(figsize=(W1, H1))
    for v, c in ((before, ORANGE), (after, BLUE)):
        ax.fill_between(B, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=tint(c, 0.6), alpha=0.6,
                        linewidth=0, zorder=2)
    mb, ma = before.mean(0), after.mean(0)
    for b, top, bot in zip(B, mb, ma):
        ax.add_patch(FancyArrowPatch((b, top - 0.03), (b, bot + 0.035), arrowstyle="-|>", mutation_scale=6,
                                     color=GRAY, linewidth=0.8, zorder=3))
    handles = [series(ax, B, mb, ORANGE, "Before averaging"), series(ax, B, ma, BLUE, "After averaging")]
    ax.set_xlim(1.5, 12.5)
    ax.set_ylim(0, 1.9)
    ax.set_xticks(B)
    ax.set_xlabel("Block")
    ax.set_ylabel("Drift From Frozen Stream")
    legend(ax, handles=handles, loc="upper left")
    fig.savefig(OUT / "dd_contraction.png")
    plt.close(fig)


def fig_alignment(pool: dict) -> None:
    rng = np.random.default_rng(0)
    grid = np.linspace(-0.12, 0.30, 400)
    modes = ("min", "max", "dropout", "mean")
    fig, ax = plt.subplots(figsize=(W1, 3.9))
    for row, mode in enumerate(modes):
        lab, c = MODE_STYLE[mode]
        v = pool[f"{mode}_cos"]
        dens = np.exp(-0.5 * ((grid[:, None] - v[None]) / 0.012) ** 2).sum(1)
        dens = 0.38 * dens / dens.max()
        ax.fill_between(grid, row + 0.06, row + 0.06 + dens, color=tint(c, 0.55), alpha=0.9, linewidth=0, zorder=2)
        ax.plot(grid, row + 0.06 + dens, color=c, linewidth=1.2, zorder=3)
        ax.scatter(v, row - 0.17 + rng.uniform(-0.09, 0.09, len(v)), s=3.5, color=c, alpha=0.35, linewidths=0,
                   zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [row - 0.02] * 2, color=c, linewidth=3.2, solid_capstyle="round", zorder=4)
        ax.scatter([med], [row - 0.02], s=16, color="white", edgecolors=c, linewidths=1.2, zorder=5)
        ax.text(0.292, row + 0.1, f"{v.mean():.3f}", color=c, ha="right", va="bottom", fontweight="bold")
    ax.axvline(0, color=GRAY, linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=1)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([MODE_STYLE[m][0] for m in modes])
    ax.set_ylim(-0.42, len(modes) - 0.45)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel("Gradient Alignment (cos θ)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "pooling_alignment.png")
    plt.close(fig)


def fig_concentration(pool: dict) -> None:
    x = np.arange(1, 769) / 768 * 100
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot([0, 100], [0, 100], color=GRAY, linewidth=1.1, linestyle=(0, (3.5, 2.5)), label="Uniform credit", zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    gradient_fill(ax, x, 100 * pool["mean_cum"], BLUE, base=0, top_alpha=0.18)
    for mode in ("mean", "dropout", "max", "min"):
        lab, c = MODE_STYLE[mode]
        y = 100 * pool[f"{mode}_cum"]
        if mode == "min":
            ax.plot(x, y, color=c, linestyle=(0, (3, 2)), linewidth=1.4, label=lab, zorder=5)
        else:
            ax.plot(x, y, color=c, linewidth=2.4 if mode == "max" else 1.7, label=lab, zorder=3, path_effects=HALO)
    knee = x[np.argmax(pool["max_cum"] > 0.999)]
    ax.annotate(f"{knee:.0f}% of dims carry\nall of the gradient", xy=(knee, 99), xytext=(16, 70), color=SUB,
                arrowprops=dict(arrowstyle="-|>", color=SUB, linewidth=0.7, mutation_scale=6,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xlabel("Top Dimensions (%)")
    ax.set_ylabel("Cumulative Gradient Mass (%)")
    legend(ax, loc="lower right")
    fig.savefig(OUT / "pooling_concentration.png")
    plt.close(fig)


def fig_seeds() -> None:
    rng = np.random.default_rng(1)
    modes = ("max", "min", "none", "mean", "dropout")
    dd = np.array([hm(r) for r in SEED_RUNS["dd"]])
    fig, ax = plt.subplots(figsize=(W1 + 0.8, H1))
    ax.axhspan(dd.mean() - dd.std(ddof=1), dd.mean() + dd.std(ddof=1), color=tint(GRAY, 0.6), alpha=0.45,
               linewidth=0, zorder=1)
    ax.axhline(dd.mean(), color=GRAY, linewidth=1.1, linestyle=(0, (3.5, 2.5)), zorder=2)
    ax.text(len(modes) - 0.45, dd.mean() + 0.06, "Drift Damping, no CGAS", color=SUB, ha="right", va="bottom")
    for i, mode in enumerate(modes):
        lab, c = MODE_STYLE[mode]
        v = np.array([hm(r) for r in SEED_RUNS[mode] if (run_dir(r) / "results.json").exists()])
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(v)), v, s=22, facecolor="white", edgecolor=c, linewidth=1.2,
                   zorder=3)
        m = v.mean()
        if len(v) > 1:
            s = v.std(ddof=1)
            ax.plot([i + 0.26, i + 0.26], [m - s, m + s], color=c, linewidth=1.6, zorder=4)
            ax.plot([i + 0.26], [m], linestyle="", zorder=5, **{**marker_kw(c), "markersize": 6.5})
            ax.text(i + 0.36, m, f"{m:.2f}", color=c, va="center", fontweight="bold")
        else:
            ax.text(i + 0.2, m, f"{m:.2f}\n(1 seed)", color=c, va="center", fontweight="bold")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_STYLE[m][0] for m in modes])
    ax.set_xlim(-0.5, len(modes) - 0.15)
    ax.set_ylabel("HM (%)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=SUB,
                      markeredgewidth=1.2, markersize=5, label="Single seed"),
               Line2D([], [], color=SUB, **{**marker_kw(SUB), "markersize": 6}, label="Mean ± std")]
    legend(ax, handles=handles, loc="lower right")
    fig.savefig(OUT / "pooling_seeds.png")
    plt.close(fig)


def fig_dropout(pool: dict, cap: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})

    n, c = pool["dropout_masks"], pool["dropout_masks_cos"]
    ax1.axhline(pool["mean_cos"].mean(), color=BLUE, linewidth=1.3, linestyle=(0, (3.5, 2.5)), zorder=2)
    ax1.set_xscale("log", base=2)
    ax1.set_xlim(0.8, 80)
    ax1.set_ylim(0.09, 0.14)
    gradient_fill(ax1, n, c, MAGENTA, base=0.09)
    handles = [series(ax1, n, c, MAGENTA, "Dropout, averaged over masks"),
               Line2D([], [], color=BLUE, linewidth=1.3, linestyle=(0, (3.5, 2.5)), label="Mean pooling")]
    ax1.set_xticks(n)
    ax1.set_xticklabels([str(int(k)) for k in n])
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xlabel("Masks Averaged")
    ax1.set_ylabel("Gradient Alignment (cos θ)")
    legend(ax1, handles=handles, loc="lower right")

    acc, ratio = cap["train_acc"], cap["ratio"]
    ax2.plot(acc, ratio, color=tint(GRAY, 0.3), zorder=3, path_effects=HALO)
    ax2.plot(acc, ratio, linestyle="", zorder=3.5, **marker_kw(tint(GRAY, 0.15)))
    for a, r, w in zip(acc, ratio, cap["width"]):
        ax2.annotate(f"{int(w)}", (a, r), xytext=(5, 3), textcoords="offset points", color=SUB, fontsize=7)
    ax2.plot([pool["dropout_train_acc"]], [pool["dropout_ratio"]], linestyle="", zorder=5,
             **{**marker_kw(MAGENTA), "markersize": 7.5})
    ax2.annotate("", xy=(pool["dropout_train_acc"] + 0.6, pool["dropout_ratio"] - 0.006),
                 xytext=(pool["mean_train_acc"] - 0.5, pool["mean_ratio"] + 0.006),
                 arrowprops=dict(arrowstyle="-|>", color=MAGENTA, linewidth=1.0, mutation_scale=7,
                                 connectionstyle="arc3,rad=0.2"), zorder=4)
    ax2.set_xlabel("CGAS Head Train Acc. (%)")
    ax2.set_ylabel("Backbone / Head Grad.")
    legend(ax2, handles=[Line2D([], [], color=tint(GRAY, 0.3), label="Mean, by input width", **marker_kw(tint(GRAY, 0.15))),
                         Line2D([], [], color=MAGENTA, linestyle="", label="Dropout, width 64",
                                **{**marker_kw(MAGENTA), "markersize": 6.5})], loc="upper right")
    fig.savefig(OUT / "dropout_mechanism.png")
    plt.close(fig)


def fig_capacity(cap: dict) -> None:
    width = cap["width"]
    seeded = {}
    for w in width:
        runs = [WIDTH_RUNS[int(w)]] + [f"{WIDTH_RUNS[int(w)]}_s{s}" for s in (1, 2)]
        runs = [r for r in runs if (run_dir(r) / "results.json").exists()]
        if len(runs) > 1:
            seeded[int(w)] = [hm(r) for r in runs]
    panels = [("Backbone / Head Grad.", cap["ratio"], BLUE), ("CGAS Novel Acc. (%)", cap["cgas_novel"], ORANGE),
              ("HM (%)", cap["hm"], AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.38})
    for ax, (lab, y, c) in zip(axes, panels):
        ax.axvspan(52, 79, color=tint(BLUE, 0.8), alpha=0.55, linewidth=0, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xlim(12, 1000)
        lo, hi = min(y), max(y)
        if lab.startswith("HM"):
            lo, hi = min(lo, *(min(v) for v in seeded.values())), max(hi, *(max(v) for v in seeded.values()))
        pad = (hi - lo) * 0.18
        ax.set_ylim(lo - pad, hi + pad * (3.4 if lab.startswith("HM") else 1))
        gradient_fill(ax, width, y, c, base=lo - pad)
        series(ax, width, y, c)
        ax.set_xticks(width)
        ax.set_xticklabels([str(int(w)) for w in width])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylabel(lab)
        ax.set_xlabel("CGAS Input Width")
    for w, v in seeded.items():
        axes[2].plot([w, w], [min(v), max(v)], color=AQUA, linewidth=1.2, zorder=3)
        axes[2].scatter([w] * len(v), v, s=9, facecolor="white", edgecolor=AQUA, linewidth=0.9, zorder=3.5)
    legend(axes[2], handles=[Line2D([], [], color=AQUA, label="Seed 0", **marker_kw(AQUA)),
                             Line2D([], [], color=AQUA, marker="o", markersize=3.5, markerfacecolor="white",
                                    linewidth=1.2, label="3-seed range")], loc="upper right")
    fig.savefig(OUT / "capacity_bottleneck.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="regenerate docs/cgas.md figures")
    ap.add_argument("--recompute", action="store_true", help="ignore cached data in runs/analysis/cgas")
    args = ap.parse_args()
    torch.set_num_threads(1)
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = cached("pooling", pooling_data, args.recompute)
    cap = cached("capacity", capacity_data, args.recompute)
    fig_fingerprint(pool)
    fig_alignment(pool)
    fig_concentration(pool)
    fig_dropout(pool, cap)
    fig_capacity(cap)
    fig_contraction(cached("contraction", contraction_data, args.recompute))
    fig_impulse(cached("impulse", impulse_data, args.recompute))
    fig_seeds()
    for mode in POOL_RUNS:
        print(f"{mode:7s} cos={pool[f'{mode}_cos'].mean():.3f} zero={float(pool[f'{mode}_fraczero']):.3f} "
              f"top10={float(pool[f'{mode}_top10']):.3f} ratio={float(pool[f'{mode}_ratio']):.3f} "
              f"|dL/dh|={float(pool[f'{mode}_grad_norm']):.5f} head train loss={float(pool[f'{mode}_train_loss']):.3f} "
              f"acc={float(pool[f'{mode}_train_acc']):.1f} | base {float(pool[f'{mode}_base_final']):.2f} "
              f"novel {float(pool[f'{mode}_novel_final']):.2f}")
    print("dropout masks cos:", np.round(pool["dropout_masks_cos"], 3), "E[mask] vs mean-pool:",
          round(float(pool["dropout_vs_mean_cos"]), 3), "mask-to-mask:", round(float(pool["dropout_mask_to_mask_cos"]), 3),
          "eval-pool loss/acc:", round(float(pool["dropout_eval_loss"]), 3), round(float(pool["dropout_eval_acc"]), 1))
    print("capacity train acc:", np.round(cap["train_acc"], 1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
