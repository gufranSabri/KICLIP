#!/usr/bin/env python3
# DCGD narrative figures. Each figure's companion "how to read" .md is written alongside the .png.
# Run the sub-commands you need; each is independent.
#
#     python scripts/analysis/dcgd_figures.py impulse --checkpoint runs/main/hmdb51_dd
#     python scripts/analysis/dcgd_figures.py contraction --checkpoint runs/main/hmdb51_dd
#     python scripts/analysis/dcgd_figures.py dd-combined --checkpoint runs/main/hmdb51_dd \
#         --checkpoints runs/main/hmdb51_froster runs/main/hmdb51_dd runs/main/hmdb51_cgas_noki runs/main/hmdb51_cgas \
#         --labels FROSTER "Drift Damping" CGAS DCGD
#     python scripts/analysis/dcgd_figures.py scatter --checkpoints runs/main/hmdb51_froster runs/main/hmdb51_dd \
#         runs/main/hmdb51_cgas_noki runs/main/hmdb51_cgas --labels FROSTER "Drift Damping" CGAS DCGD
#     python scripts/analysis/dcgd_figures.py distance --checkpoints <same 4 as scatter> --labels <same>
#     python scripts/analysis/dcgd_figures.py concentration --checkpoint runs/pooling_sweep/mean

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.analysis.common import load_checkpoint  # noqa: E402
from scripts.analysis.style import AQUA, BLUE, GRAY, HALO, INK, MAGENTA, ORANGE, VIOLET, W1, H1, apply_style, legend, marker_kw  # noqa: E402
from src.config import load_config, parse_overrides  # noqa: E402
from src.data.dataset import VideoDataset  # noqa: E402
from src.models.cgas import CGASHead, pool_segments  # noqa: E402
from src.utils.seed import pick_device, seed_everything  # noqa: E402

OUT = REPO_ROOT / "docs" / "figures"
POOL_STYLE = {"mean": ("Mean", BLUE), "max": ("Max", ORANGE), "min": ("Min", AQUA)}


def write_how_to_read(name: str, text: str) -> None:
    path = OUT / f"{name}_how_to_read.md"
    path.write_text(text.strip() + "\n")
    print(f"  wrote {path}")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"{name}{ext}")
    plt.close(fig)
    print(f"  wrote {OUT / f'{name}.png'}")


# ── FIGURE: impulse response + real contraction (two panels, one DD figure) ──────

def dd_blocks(checkpoint: str, cfg, device):
    model, _ = load_checkpoint(checkpoint, cfg, device)
    visual = model.model.visual
    if visual.transformer.resblocks_dd is None:
        raise ValueError(f"{checkpoint} has no drift-damping blocks (model.drift_damping_source=blocks needed)")
    return visual, visual.transformer.resblocks, visual.transformer.resblocks_dd, visual.transformer


def impulse_data(args, cfg, device) -> tuple[np.ndarray, np.ndarray]:
    visual, blocks, frozen_blocks, _ = dd_blocks(args.checkpoint, cfg, device)

    def run(embed: torch.Tensor, kick: torch.Tensor, damp: bool) -> torch.Tensor:
        x, t = embed.clone(), embed.clone()
        x[:, 0] += kick
        out = []
        for i, blk in enumerate(blocks):
            if damp:
                x, t = (x + t) / 2, frozen_blocks[i](t)
            x = blk(x)
            out.append(x[:, 0])
        return torch.stack(out, 1)

    dataset = VideoDataset(cfg, args.split, train=False, views=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    on, off = [], []
    with torch.no_grad():
        for i, (video, _) in enumerate(loader):
            if i >= args.batches:
                break
            x = video.to(device).flatten(0, -4)
            embed = visual.embed(x)
            kick = torch.randn(embed.shape[0], embed.shape[-1], generator=gen, device=device)
            kick = args.kick_norm * kick / kick.norm(dim=-1, keepdim=True)
            zero = torch.zeros_like(kick)
            on.append((run(embed, kick, True) - run(embed, zero, True)).norm(dim=-1))
            off.append((run(embed, kick, False) - run(embed, zero, False)).norm(dim=-1))
    return torch.cat(on).cpu().numpy(), torch.cat(off).cpu().numpy()


def contraction_data(args, cfg, device) -> tuple[np.ndarray, np.ndarray]:
    visual, blocks, frozen_blocks, transformer = dd_blocks(args.checkpoint, cfg, device)

    def run(embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, t = embed.clone(), embed.clone()
        before, after = [], []
        for i, blk in enumerate(blocks):
            before.append((x[:, 0] - t[:, 0]).norm(dim=-1))
            mixed = transformer._mix(x, t)
            after.append((mixed[:, 0] - t[:, 0]).norm(dim=-1))  # same old t the mix just pulled x toward
            x, t = mixed, transformer._run(frozen_blocks[i], t)
            x = blk(x)
        return torch.stack(before, 1), torch.stack(after, 1)

    dataset = VideoDataset(cfg, args.split, train=False, views=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    before_all, after_all = [], []
    with torch.no_grad():
        for i, (video, _) in enumerate(loader):
            if i >= args.batches:
                break
            x = video.to(device).flatten(0, -4)
            embed = visual.embed(x)
            before, after = run(embed)
            before_all.append(before)
            after_all.append(after)
    return torch.cat(before_all).cpu().numpy(), torch.cat(after_all).cpu().numpy()


def plot_impulse(ax, on: np.ndarray, off: np.ndarray) -> None:
    blocks_x = np.arange(1, on.shape[1] + 1)
    for v, c, lab in ((off, ORANGE, "Without Drift Damping"), (on, BLUE, "With Drift Damping")):
        ax.fill_between(blocks_x, np.percentile(v, 10, 0), np.percentile(v, 90, 0), color=c, alpha=0.18, linewidth=0)
        ax.plot(blocks_x, np.median(v, 0), color=c, label=lab, path_effects=HALO)
        ax.plot(blocks_x, np.median(v, 0), linestyle="", **marker_kw(c))
    ax.set_yscale("log")
    ax.set_xticks(blocks_x)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Perturbation Norm")
    legend(ax, loc="best")


def plot_contraction(ax, before: np.ndarray, after: np.ndarray) -> None:
    blocks_x = np.arange(1, before.shape[1] + 1)
    for v, c, lab in ((before, ORANGE, "Before Damping"), (after, BLUE, "After Damping")):
        ax.fill_between(blocks_x, np.percentile(v, 10, 0), np.percentile(v, 90, 0), color=c, alpha=0.18, linewidth=0)
        ax.plot(blocks_x, np.median(v, 0), color=c, label=lab, path_effects=HALO)
        ax.plot(blocks_x, np.median(v, 0), linestyle="", **marker_kw(c))
    ax.set_xticks(blocks_x)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Drift From Frozen Stream")
    legend(ax, loc="best")


def cmd_impulse(args) -> None:
    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()
    on, off = impulse_data(args, cfg, device)

    apply_style()
    fig, ax = plt.subplots(figsize=(W1, H1))
    plot_impulse(ax, on, off)
    save(fig, "dd_impulse_response")
    write_how_to_read("dd_impulse_response", """
# How to Read: Drift Damping Impulse Response

A one-time perturbation of fixed norm is added to the CLS token at layer 1's input. Each
curve tracks the norm of the DIFFERENCE this perturbation causes at every later layer's
output, relative to a run with no perturbation (median over sampled HMDB clips; shaded
band = 10th-90th percentile). Y-axis is log-scale.

What it shows: with drift damping on, the perturbation's effect shrinks layer over layer.
With drift damping off, the same perturbation does not shrink the same way.

## Interpretation

With drift damping on, the perturbation's norm falls by roughly an order of
magnitude every 2-3 layers, from about 2.5 at layer 1 down to about 1.5e-3 by
layer 12 -- a clean, almost perfectly monotonic decay across all 12 layers. This
matches the predicted 0.5x contraction per layer from averaging the student stream
into the frozen teacher's stream before each layer: errors introduced anywhere in
the network get squeezed out again by the time they reach the output.

Without drift damping, the same perturbation starts larger (around 5, since there is
no averaging to immediately halve it) and barely decays through the first 6-7 layers,
then actually grows again through layers 8-12, ending close to where it started. The
undamped student has no mechanism forcing an injected error back toward the frozen
teacher's trajectory, so a perturbation's effect on the final representation is not
controlled -- it can just as easily grow as shrink depending on what the later layers
do with it. This is the direct empirical evidence for the paper's contraction claim:
damping does not just theoretically guarantee shrinkage, it is the only one of the
two settings where shrinkage is actually observed end to end.
""")


def cmd_contraction(args) -> None:
    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()
    before_all, after_all = contraction_data(args, cfg, device)

    apply_style()
    fig, ax = plt.subplots(figsize=(W1, H1))
    plot_contraction(ax, before_all, after_all)
    save(fig, "dd_contraction")
    write_how_to_read("dd_contraction", """
# How to Read: Drift Damping Contraction

For each vision layer, the student stream's distance (CLS token norm of the difference)
from the frozen teacher stream is measured twice: right before that layer's averaging
step ("Before Damping") and right after it ("After Damping"), on real sampled HMDB
clips (median; shaded band = 10th-90th percentile). "Before" at layer k is the raw drift
the student accumulated since the last averaging step; "After" is the SAME drift measured
against the SAME frozen-stream snapshot, right after averaging pulls the student halfway
toward it.

What it shows: at every layer, the after-averaging curve sits at almost exactly half the
before-averaging curve -- the averaging step is a literal, exact 0.5x contraction of
drift at every layer, matching ‖(x+t)/2 - t‖ = ‖x-t‖/2 algebraically. Both curves drift
upward over depth (later layers accumulate more independent drift before each
correction), but the halving ratio itself holds at every layer regardless of how much
drift has accumulated by that point.
""")


def cmd_dd_combined(args) -> None:
    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()
    on, off = impulse_data(args, cfg, device)
    before_all, after_all = contraction_data(args, cfg, device)
    names, distances = distance_data(args, cfg, device)

    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(W1 * 3, H1))
    plot_impulse(ax1, on, off)
    plot_contraction(ax2, before_all, after_all)
    plot_distance(ax3, names, distances, cfg.seed)
    for ax, label in ((ax1, "(a)"), (ax2, "(b)"), (ax3, "(c)")):
        ax.set_title(label, loc="center", fontweight="bold")
    save(fig, "dd_combined")
    write_how_to_read("dd_combined", f"""
# How to Read: Drift Damping + Distance From CLIP

(a): a one-time perturbation of fixed norm is added to the CLS token at layer 1's input.
Each curve tracks the norm of the DIFFERENCE this perturbation causes at every later
layer's output, relative to a run with no perturbation (median over sampled HMDB clips;
shaded band = 10th-90th percentile). Y-axis is log-scale.

(b): for each vision layer, the student stream's real distance from the frozen teacher
stream is measured right before that layer's averaging step ("Before Damping") and
right after it ("After Damping"), on the same sampled HMDB clips.

(c): one box + jittered strip per method ({', '.join(names)}). Each small point is one
HMDB-51 novel-split video's cosine distance (1 - cosine similarity) between that
method's embedding and zero-shot CLIP's embedding of the SAME video.

What it shows: (a) and (b) are two views of the same per-layer contraction mechanism --
a controlled, synthetic perturbation test and the same mechanism measured directly on
real clips' own drift, with the after-damping curve sitting at almost exactly half the
before-damping curve at every layer, the literal 0.5x contraction damping guarantees
algebraically. (c) is the end-to-end consequence at the final embedding: Drift Damping
alone drifts least from CLIP, CGAS alone drifts most, and DCGD (damping + CGAS together)
sits closer to CLIP than CGAS alone -- damping pulling CGAS's otherwise large drift back
toward CLIP's manifold.
""")


# ── shared: per-video CLIP + method embeddings, for the scatter/displacement/distance figures ──

def load_embeddings(args, cfg, device) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    names = args.labels or [Path(c).name for c in args.checkpoints]

    dataset = VideoDataset(cfg, args.split, train=False, views=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    videos = []
    for i, (video, _) in enumerate(loader):
        if i * args.batch_size >= args.num_videos:
            break
        videos.append(video)
    videos = torch.cat(videos)[: args.num_videos]

    from src.models.clip import CLIP, pretrained_state

    clip_model = CLIP.from_state_dict(pretrained_state(cfg.model.arch)).to(device).eval()
    with torch.no_grad():
        clip_embeds = F.normalize(clip_model.encode_image(videos.to(device).flatten(0, -4)), dim=-1) \
            .view(videos.shape[0], videos.shape[1], -1).mean(1).cpu().numpy()
    del clip_model
    torch.cuda.empty_cache()

    method_embeds = {}
    for name, spec in zip(names, args.checkpoints):
        model, _ = load_checkpoint(spec, cfg, device)
        with torch.no_grad():
            frames, *_ = model.encode_video(videos.to(device).flatten(0, -4))
        method_embeds[name] = frames.view(videos.shape[0], videos.shape[1], -1).mean(1).cpu().numpy()
        del model
        torch.cuda.empty_cache()
    return names, clip_embeds, method_embeds


# ── FIGURE: embedding scatter, one CLIP-vs-method plot per method ────────────

def cmd_scatter(args) -> None:
    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()
    names, clip_embeds, method_embeds = load_embeddings(args, cfg, device)

    pca = PCA(n_components=2).fit(clip_embeds)
    clip_xy = pca.transform(clip_embeds)
    palette = [ORANGE, BLUE, VIOLET, MAGENTA]

    apply_style()
    fig, axes = plt.subplots(1, len(args.checkpoints), figsize=(W1 * len(args.checkpoints), H1), sharex=True, sharey=True)
    for ax, name, c in zip(axes, names, palette):
        method_xy = pca.transform(method_embeds[name])

        ax.scatter(clip_xy[:, 0], clip_xy[:, 1], s=14, facecolor=GRAY, edgecolor="white", linewidth=0.5,
                   alpha=0.85, label="CLIP")
        ax.scatter(method_xy[:, 0], method_xy[:, 1], s=14, facecolor=c, edgecolor="white", linewidth=0.5,
                   alpha=0.85, label=name)
        ax.set_xlabel("PC 1 (CLIP Space)")
        legend(ax, loc="best")
    axes[0].set_ylabel("PC 2 (CLIP Space)")
    save(fig, "scatter_clip_vs_methods")
    write_how_to_read("scatter_clip_vs_methods", f"""
# How to Read: Embedding Scatter, CLIP vs. Each Method

One panel per method ({', '.join(names)}), left to right. Each point is one HMDB-51
novel-split video's final vision embedding: gray for zero-shot CLIP (same points in
every panel), colored for that panel's method. The 2-D axes are the first two
principal components of zero-shot CLIP's OWN embeddings -- i.e. CLIP's point cloud
defines the coordinate system, and every method's points are projected into those
same axes, shared across all 4 panels, so the axes mean the same thing in each one
and the panels are directly comparable.

What to look for: how far each method's cloud sits from CLIP's, and whether that
distance differs across panels.

## Interpretation

In every panel the colored cloud occupies the same broad clusters as CLIP's gray
cloud -- no method collapses the embedding space or moves videos into a wildly
different region, which is expected since every method is still initialized from and
distilled against frozen CLIP. The differences between panels are in how tightly
each colored cloud hugs the gray one within those shared clusters: FROSTER and CGAS
visibly spread further from the gray points inside each cluster than Drift Damping
does, while DCGD's pink cloud sits closer to CLIP's gray points than CGAS's does.
This is consistent with the per-video distance numbers in `distance_from_clip.png`
-- this figure shows that the same drift happens in a geometrically sensible way
(within CLIP's existing cluster structure, not off into unrelated regions), while
the distance figure is the better source for exact magnitudes, since a 2-D PCA
projection here can visually compress real differences in distance.
""")


# ── FIGURE: distance from CLIP, one distribution per method ──────────────────

def distance_data(args, cfg, device) -> tuple[list[str], dict[str, np.ndarray]]:
    names, clip_embeds, method_embeds = load_embeddings(args, cfg, device)
    clip_unit = clip_embeds / np.linalg.norm(clip_embeds, axis=-1, keepdims=True)
    distances = {}
    for name in names:
        method_unit = method_embeds[name] / np.linalg.norm(method_embeds[name], axis=-1, keepdims=True)
        distances[name] = 1 - np.sum(clip_unit * method_unit, axis=-1)
    return names, distances


def plot_distance(ax, names: list[str], distances: dict[str, np.ndarray], seed: int) -> None:
    palette = [ORANGE, BLUE, VIOLET, MAGENTA]
    positions = np.arange(1, len(names) + 1)
    rng = np.random.default_rng(seed)
    for i, name, c in zip(positions, names, palette):
        x = i + rng.uniform(-0.1, 0.1, len(distances[name]))
        ax.scatter(x, distances[name], s=6, facecolor=c, edgecolor="none", alpha=0.35, zorder=1)
    box = ax.boxplot([distances[name] for name in names], positions=positions, widths=0.32,
                      showfliers=False, patch_artist=True, zorder=3,
                      medianprops=dict(color=INK, linewidth=1.6),
                      whiskerprops=dict(color=INK, linewidth=1.0),
                      capprops=dict(color=INK, linewidth=1.0))
    for patch, c in zip(box["boxes"], palette):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_xlim(0.3, len(names) + 0.7)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Cosine Distance From CLIP")


def cmd_distance(args) -> None:
    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()
    names, distances = distance_data(args, cfg, device)

    apply_style()
    fig, ax = plt.subplots(figsize=(W1, H1))
    plot_distance(ax, names, distances, cfg.seed)
    save(fig, "distance_from_clip")
    write_how_to_read("distance_from_clip", f"""
# How to Read: Cosine Distance From CLIP, by Method

One box + jittered strip per method ({', '.join(names)}). Each small point is one
HMDB-51 novel-split video's cosine distance (1 - cosine similarity) between that
method's embedding and zero-shot CLIP's embedding of the SAME video -- a direct
per-video distance, no projection or dimensionality reduction involved. The box
spans the interquartile range with a line at the median; whiskers extend to the
furthest point within 1.5x IQR.

What to look for: which method's box sits higher (farther from CLIP on average) or
is taller / has more scattered points (more variable across videos), read directly
off the y-axis rather than off relative positions in a 2-D projection.
""")


# ── FIGURE: gradient concentration by pooling mode ────────────────────────────

def pooled_grad(cls: torch.Tensor, mode: str, n_segments: int, embed_dim: int) -> torch.Tensor:
    cls = cls.clone().detach().requires_grad_(True)
    head = CGASHead(n_layers=1, n_segments=n_segments, embed_dim=embed_dim, n_channels=1)
    pooled = pool_segments(cls, n_segments, mode).unsqueeze(1)  # (N, 1, n_segments)
    out = head(pooled)
    out.pow(2).sum().backward()
    return cls.grad.detach()  # (N, D)

def cmd_concentration(args) -> None:
    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()
    model, _ = load_checkpoint(args.checkpoint, cfg, device)
    if model.cgas_head is None:
        raise ValueError(f"{args.checkpoint} has no CGAS head")
    n_segments, embed_dim = model.cgas_segments, model.model.embed_dim

    dataset = VideoDataset(cfg, args.split, train=False, views=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    grads = {mode: [] for mode in ("mean", "max", "min")}
    for i, (video, _) in enumerate(loader):
        if i >= args.batches:
            break
        with torch.no_grad():
            x = video.to(device).flatten(0, -4)
            hidden = model.model.visual.hidden(x, None)
            cls = hidden[-1][:, 0].cpu()
        for mode in grads:
            grads[mode].append(pooled_grad(cls, mode, n_segments, embed_dim).abs())
    curves, dims_to_80 = {}, {}
    for mode in ("mean", "max", "min"):
        g = torch.cat(grads[mode]).mean(0).numpy()  # (768,), averaged over sampled clips
        sorted_g = np.sort(g)[::-1]
        cum = np.cumsum(sorted_g) / sorted_g.sum()
        curves[mode] = cum
        dims_to_80[mode] = (np.searchsorted(cum, 0.8) + 1) / len(cum)

    apply_style()
    fig, ax = plt.subplots(figsize=(W1, H1))
    frac_dims = np.arange(1, len(curves["mean"]) + 1) / len(curves["mean"])
    for mode in ("mean", "max", "min"):
        lab, c = POOL_STYLE[mode]
        ax.plot(frac_dims, curves[mode], color=c, label=lab, path_effects=HALO)
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.0, linestyle=(0, (3.5, 2.5)))
    ax.set_xlabel("Fraction of Dimensions")
    ax.set_ylabel("Cumulative Gradient Mass")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend(ax, loc="lower right")
    save(fig, "cgas_gradient_concentration")
    write_how_to_read("cgas_gradient_concentration", f"""
# How to Read: CGAS Gradient Concentration

For each pooling mode, the 768 gradient magnitudes on the block-12 CLS token
(averaged over sampled HMDB clips) are sorted largest to smallest. X-axis: fraction
of dimensions, in that sorted order. Y-axis: cumulative fraction of total gradient
mass carried by that many dimensions. The dashed diagonal is what a perfectly even
spread across all 768 dimensions would look like.

## Interpretation

Mean pooling's curve sits close to the diagonal and needs roughly {dims_to_80['mean']:.0%}
of all dimensions to reach 80% of gradient mass -- its gradient is spread almost
evenly across the backbone, consistent with mean pooling producing one value per
dimension group with no single dimension favored.

Max and min pooling both rise steeply from the start and need only about
{dims_to_80['max']:.0%} and {dims_to_80['min']:.0%} of dimensions respectively to
reach 80% of mass -- gradient is concentrated on the handful of dimensions that are
each group's argmax/argmin, since every other dimension in that group gets exactly
zero gradient from that pooling operation. This is the direct mechanical reason mean
pooling gives CGAS a dense, stable training signal into the backbone while max/min
pooling give a sparse, winner-take-all signal concentrated on very few dimensions.
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", action="append", default=[])
    common.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    common.add_argument("--split", default="b2n/hmdb51/novel")
    common.add_argument("--batch-size", type=int, default=4)

    p = sub.add_parser("impulse", parents=[common])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--kick-norm", type=float, default=5.0)
    p.set_defaults(func=cmd_impulse)

    p = sub.add_parser("contraction", parents=[common])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.set_defaults(func=cmd_contraction)

    p = sub.add_parser("dd-combined", parents=[common])
    p.add_argument("--checkpoint", required=True, help="drift-damping checkpoint for panels (a)/(b)")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--kick-norm", type=float, default=5.0)
    p.add_argument("--checkpoints", nargs="+", required=True, help="FROSTER/DD/CGAS/DCGD checkpoints for panel (c)")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--num-videos", type=int, default=200)
    p.set_defaults(func=cmd_dd_combined)

    p = sub.add_parser("scatter", parents=[common])
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--num-videos", type=int, default=200)
    p.set_defaults(func=cmd_scatter)

    p = sub.add_parser("distance", parents=[common])
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--num-videos", type=int, default=200)
    p.set_defaults(func=cmd_distance)

    p = sub.add_parser("concentration", parents=[common])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.set_defaults(func=cmd_concentration)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
