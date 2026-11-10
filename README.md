# KICLIP

**Knowledge Injection Improves Distillation: open-vocabulary action recognition built on FROSTER (ICLR 2024).**

A student CLIP is fine-tuned on videos while it is kept close to the original CLIP in three ways:

1. **Distillation (FROSTER)**: residual projectors on the student's frame and class features (`x + 0.1 * MLP(x)`)
   are pulled to a frozen CLIP teacher's with a cosine loss, weighted by `model.distill_weight`.
2. **Knowledge injection (KICLIP)**: a frozen copy of the pretrained vision blocks runs next to the student's,
   fed the same patch tokens. Before every student block its tokens are averaged with the frozen branch's tokens
   at the same depth (`model.inject`). It is active at test time too.
3. **Aux classifier (KICLIP)**: a closed-set classifier on the frame-averaged projected video features
   (Linear 512→768, GELU, torchvision's pretrained ImageNet ViT-B/16 head, GELU, Linear 1000→C), trained with
   cross-entropy weighted by `model.aux_weight`. Training only.

The student's vision attention also reads the keys/values of each frame's neighbours (`model.temporal: expand`),
class embeddings come from an ensemble of templated GPT-rephrased descriptions, and `averaged.pt` is the mean of
the weights after epochs `train.average_from..train.epochs`.

## Repo layout

```
kiclip.py              CLI: train / test
kiclip.slurm           batch job running every experiment for $METHODS, then the tables
src/models/clip.py     OpenAI CLIP as plain modules (names match the checkpoints), temporal attention, injection
src/models/kiclip.py   student wrapper: projectors, aux classifier, distillation loss
src/eigencam.py        EigenCAM over the vision blocks
src/data/              PyAV decoding, clip sampling and transforms, prompt templates
src/train.py           training loop (AMP, gradient accumulation, DDP, resume, online weight averaging)
src/test.py            multi-view evaluation
configs/               default.yaml (KICLIP), one yaml per (protocol, dataset), methods/*.yaml overlays
splits/                FROSTER's split CSVs and class-description JSONs
scripts/               install.sh, troubleshooting.sh, data/ (download + link), experiments/*.sh, tables/*.py,
                       figures/eigencam.py
```

## Install

```bash
bash scripts/install.sh
```

Everything is a pip wheel (on Compute Canada, from the local wheelhouse): no git installs, no SlowFast, detectron2,
pytorchvideo or fvcore. The OpenAI CLIP weights come from the HF Hub through `open_clip` (into `$HF_HOME`), and the
aux classifier's ImageNet head from torchvision (into `$TORCH_HOME`), the first time they are needed.

## Data

`data_root` (in `configs/default.yaml`) defaults to `~/scratch/ahmedubc/KICLIP-data`.

```bash
bash scripts/data/download.sh                 # UCF101, HMDB51, K400 train + val, the K600 val classes the splits use
python scripts/data/organize.py --check --workers 8 --delete-raw
```

`download.sh` fetches into `{data_root}/raw/<dataset>/` and can be rerun: finished parts are skipped. K400 comes from
the official CVDF release (~380 GB). The K600 clips the splits use are in CVDF's per-class `600/train` tarballs: they
are streamed (~240 GB) and only the ~10k listed clips are kept. UCF101 and HMDB51 come from Hugging Face copies of the
original archives (`quchenyuan/UCF101-ZIP`, `CVML-TueAI/HMDB51`). SSv2 is behind a research license. Accept it on
[Qualcomm's site](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads), put the
`20bn-something-something-v2-??` parts in `{data_root}/raw/ssv2/`, then run `download.sh ssv2`.

`organize.py` moves every video a split CSV lists from `raw/` to `{data_root}/<dataset>/` under the exact name the CSV
uses. A video listed under two names is hard-linked. Kinetics videos are matched by YouTube id, because CVDF names them
`<id>_<start>_<end>.mp4` while the val CSVs use `<id>.mp4`. `--check` decodes a frame of each video and leaves out
broken ones. `--delete-raw` then deletes `raw/<dataset>`, including videos no split lists, and marks the dataset done
so `download.sh` skips it. It prints per-split coverage and writes the unavailable videos of each split to
`{data_root}/logs/`. Missing videos are skipped with a warning at train and test time (the count is in the log and in
`n` of every result).

```
{data_root}/
    k400/train/<class>/<id>_<start>_<end>.mp4
    k400/val/<class>/<id>.mp4
    k600/val/<class>/<id>_<start>_<end>.mp4
    ucf101/<Class>/v_<Class>_gXX_cYY.avi
    hmdb51/<class>/<name>.avi
    ssv2/<id>.webm
```

FROSTER trained on VoV3D's copy of K400 with videos resized to a 256px short side; CVDF keeps the original
resolution. Inputs end up identical (short side jittered to 224–256, then a 224 crop), but decoding is slower.

## Splits

A split is `<protocol>/<dataset>/<set>`, i.e. `splits/<protocol>/<dataset>/<set>.csv`, with class descriptions in
`<set>.json` if present, else the folder's `classes.json`.

| Protocol | Train | Test |
|---|---|---|
| `b2n` | `b2n/{k400,ssv2}/train_s{1,2,3}`, `b2n/{hmdb51,ucf101}/train_s1` (16 shots of the base classes) | `b2n/<dataset>/base`, `b2n/<dataset>/novel` |
| `cross` | `cross/k400/train` (all 400 classes) | `cross/{ucf101,hmdb51}/{full,split1,split2,split3}`, `cross/k600/split{1,2,3}` |

## Methods

`configs/default.yaml` is full KICLIP. Stack a method overlay after the dataset config to switch parts off:

| Overlay | `model.inject` | `model.aux_weight` |
|---|---|---|
| `configs/methods/kiclip.yaml` | true | 1.0 |
| `configs/methods/kiclip_no_aux.yaml` | true | 0 |
| `configs/methods/kiclip_no_injection.yaml` | false | 1.0 |
| `configs/methods/froster.yaml` | false | 0 |

## Usage

```bash
python kiclip.py --config configs/b2n/hmdb51.yaml train --run-name b2n_hmdb51_s1_kiclip
python kiclip.py --config configs/b2n/hmdb51.yaml test  --checkpoint runs/b2n_hmdb51_s1_kiclip/averaged.pt
python kiclip.py --config configs/b2n/k400.yaml --config configs/methods/froster.yaml train --run-name b2n_k400_s2_froster \
    --set data.train=b2n/k400/train_s2
python kiclip.py --config configs/cross/k400.yaml test --checkpoint runs/cross_k400_kiclip/averaged.pt --splits cross/k600/split1
```

`--config` can be repeated (later files win) and `--set key=value` overrides any key, on either side of the
subcommand. `test` evaluates `data.test` unless `--splits` is given and writes `test_<split>.json` next to the
checkpoint. It also loads the original FROSTER and KICLIP releases (`swa_2_22.pth`); a checkpoint with injection
blocks needs `model.inject: true` and one without needs `false`, which `test` checks.

Multi-GPU runs go through `torchrun`; `train.batch_size` is global, so the paper's batch is kept on any GPU count by
gradient accumulation (`train.micro_batch_size` per GPU per forward):

```bash
torchrun --standalone --nproc_per_node=4 kiclip.py --config configs/cross/k400.yaml train --run-name cross_k400_kiclip
```

A run directory holds `config.json`, `train.log`, `history.json`, a code snapshot, `last.pt` while training
(rerunning the same command resumes from it) and `averaged.pt` + `results.json` at the end.

## EigenCAM

```bash
python scripts/figures/eigencam.py --split b2n/hmdb51/novel --index 0 \
    --checkpoints runs/b2n_hmdb51_s1_froster/averaged.pt runs/b2n_hmdb51_s1_kiclip/averaged.pt --labels FROSTER KICLIP
python scripts/figures/eigencam.py --video some_clip.mp4 --checkpoints clip runs/cross_k400_kiclip/averaged.pt
```

One row per checkpoint (`clip` = the pretrained CLIP), one column per vision block (`--layers`, default 5–12,
1-indexed block outputs), on the middle frame of the middle test view (`--frames`, `--view`). Each checkpoint is
rebuilt from its own weights and config, with injection detected from its weights. Maps follow pytorch-grad-cam's
EigenCAM: first principal component of the patch tokens, ReLU, min-max, bilinear upsampling, jet overlay at 0.5.
Writes `docs/figures/eigencam/<video>.png` and `.pdf`.

## Experiments

```bash
bash scripts/experiments/b2n.sh              # every method x dataset x seed: train, then test base + novel
bash scripts/experiments/cross_dataset.sh    # every method: K400 -> UCF101 / HMDB51 / K600
python scripts/tables/b2n.py                 # docs/tables/b2n.md, per method, with FROSTER's paper numbers
python scripts/tables/cross_dataset.py       # docs/tables/cross_dataset.md
mkdir -p slurm && sbatch kiclip.slurm        # all of the above on 4 L40S
```

Both scripts skip finished work (`RERUN=1` forces it), run `METHODS="kiclip froster"` by default (any overlay name
works) and take `LAUNCH="torchrun --standalone --nproc_per_node=N"`.

## Differences from the original code

- Clip sampling is done in seconds: `num_frames` samples evenly over `num_frames * sampling_rate / target_fps`
  seconds. SlowFast's PyAV path spreads the same 8 samples over a slightly longer window and clamps the last ones.
- The optional per-epoch validation (`train.eval_every`) uses one centre view; it is not used for model selection.
- Weight averaging is a running mean during training instead of averaging saved per-epoch checkpoints.
- SSv2 uses its last-epoch weights (`average_from: 12`), as FROSTER's SSv2 test script does.
- The KICLIP reference scripts trained with a batch of 4 on one GPU; here every method uses FROSTER's batch sizes
  (32, or 64 for UCF101 and cross-dataset K400) so the methods are compared at equal batch.

Not ported, because no FROSTER or KICLIP experiment uses it: the SlowFast model zoo, AVA/Charades/ImageNet,
MixUp/RandAugment, EWC, expert routing, weight patching, linear-connectivity climbing, prompt tuning, the ViL/SCAR
side models, submitit and TensorBoard.

## Citation

```bibtex
@inproceedings{huang2024froster,
  title={FROSTER: Frozen CLIP is a Strong Teacher for Open-Vocabulary Action Recognition},
  author={Xiaohu Huang and Hao Zhou and Kun Yao and Kai Han},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
