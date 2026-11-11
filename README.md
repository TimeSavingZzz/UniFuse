# UniFuse

[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

Code for the paper **"UniFuse: A Controlled Testbed and Practical Guidelines for
Feature Fusion in Image Restoration"** (under review, *Applied Intelligence*).

UniFuse holds the backbone, auxiliary encoder, loss, optimizer, schedule and data
pipeline fixed, and varies only the module that injects auxiliary features into the
decoder. The repository contains the training, evaluation and analysis code behind
the tables and figures in the paper: two backbones, four datasets, five fusion
mechanisms, one capacity control and a no-fusion baseline, about 80 training runs.

---

## Configurations

| Configuration | Role | Operation |
|---|---|---|
| Concat | fusion mechanism | channel concatenation followed by a 1x1 convolution |
| CrossAttn | fusion mechanism | multi-head cross-attention; queries from the decoder, keys and values from the auxiliary encoder |
| FiLM | fusion mechanism | channel-wise scale and shift predicted from auxiliary features |
| Gated | fusion mechanism | sigmoid gate between decoder and auxiliary features |
| ASF | fusion mechanism | channel-wise mixture of a FiLM branch and a gated branch, with a learned mixing weight |
| Large | capacity control | Concat with the backbone widened (Restormer dim 48 -> 64, NAFNet width 16 -> 32) |
| No fusion | baseline | auxiliary branch removed |

The auxiliary encoder is a shared degradation encoder (SDE): it takes a grayscale
copy of the degraded input, produces features at three scales, and shares weights
across tasks. Features from the second and third scales are injected at decoder
levels D2 and D3; the first scale is not used.

---

## Datasets

| Dataset | Task | Data used | Source |
|---|---|---|---|
| SD7K | document shadow removal | 200-pair training subset, 190-pair test subset (256x256) | https://github.com/CXH-Research/DocShadow-SD7K |
| RDD | document shadow removal | 4,371 train / 545 test pairs, 256x256 crops | https://github.com/leiyingtie/DocDeshadower |
| LOL-v1 | low-light enhancement | 485 train / 15 test pairs | https://github.com/weichen582/RetinexNet |
| Rain100H | deraining | 1,800 train / 100 test pairs | https://github.com/nnUyi/DerainZoo |

Expected directory layout:

```
dataset/
├── SD7K/{train,test}/{input,target}/
├── RDD/{train,test}/                # img/, gt/, back_gt/ as released
├── LOL-v1/{train,test}/{input,target}/
└── Rain100H/{train,test}/{input,target}/
```

The input and target subdirectory of each dataset are defined in `DATASET_CONFIGS`
in `train_compare_models.py`, so a different RDD target layout can be selected there.

---

## Repository layout

```
models/
├── comparison_models.py     # Restormer, NAFNet and the ShadowGuided* fusion variants
├── fusion_models.py         # NAFNet fusion variants and the ASF module
├── model.py                 # original ShaDocFormer backbone (shared component)
├── text_aware_model.py      # text-aware variant (shared component)
├── mask.py, refine.py, cnn_backbone.py
data/
├── dataset_RGB.py           # SD7K / RDD loader (grayscale auxiliary input)
├── dataset_lol.py           # LOL-v1 loader
├── dataset_rain100h.py      # Rain100H loader
├── data_RGB.py, synthetic_dataset.py
config/config.py             # training configuration
train_compare_models.py      # main training entry point, all configurations
train.py, test.py            # original ShaDocFormer training/eval scripts
evaluate_checkpoint.py       # evaluate one checkpoint
eval_all_checkpoints.py      # evaluate a directory of checkpoints
eval_efficiency.py           # parameters, FLOPs, inference time, memory
eval_lpips.py                # LPIPS
eval_ocr.py                  # OCR-based text preservation
eval_cross_dataset.py        # zero-shot cross-dataset transfer
cross_dataset_eval.py        # cross-dataset result aggregation
run_scaling_curve.py         # data-volume scaling experiment
run_scaling_chain.py         # sequential launcher for the scaling runs
gen_scaling_fig.py           # scaling-curve figure
cka_analysis.py              # CKA between encoder and decoder features
alpha_histogram.py           # ASF mixing-weight histogram
visualize_attention.py       # decoder self-attention maps
run_attn_viz.py              # batch attention visualization
plot_training_curves.py      # training curves from logs
environment.yml, requirements.txt
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate shadocformer
```

The environment uses PyTorch 2.0.1 with CUDA 11.8 and was tested on 4x RTX 3090
(24 GB).

---

## Training

All runs use AdamW (beta1=0.9, beta2=0.999, weight decay 1e-4), initial learning
rate 2e-4 with cosine annealing, and 200 epochs. Document shadow removal uses
L1 + SSIM + FFT loss; low-light enhancement and deraining use L1.

| Configuration | Restormer `--model` | NAFNet `--model` |
|---|---|---|
| No fusion | `restormer` | `nafnet` |
| Concat | `shadow_guided_restormer_concat` | `shadow_guided_nafnet_concat` |
| CrossAttn | `shadow_guided_restormer_crossattn` | `shadow_guided_nafnet_crossattn` |
| FiLM | `shadow_guided_restormer_film` | `shadow_guided_nafnet_film` |
| Gated | `shadow_guided_restormer_gated` | `shadow_guided_nafnet_gated` |
| ASF | `shadow_guided_restormer_asf` | `shadow_guided_nafnet_asf` |
| Large | `shadow_guided_restormer_large` | `shadow_guided_nafnet_large` |

```bash
# Restormer, Concat, SD7K
python train_compare_models.py --model shadow_guided_restormer_concat \
    --dataset sd7k --epochs 200 --res 256 --output ./experiment_results/restormer_concat_sd7k/

# NAFNet, Gated, LOL-v1
python train_compare_models.py --model shadow_guided_nafnet_gated \
    --dataset lol --epochs 200 --res 256 --output ./experiment_results/nafnet_gated_lol/

# FiLM on RDD uses gradient clipping (v1 diverged to NaN at epochs 125 and 130)
python train_compare_models.py --model shadow_guided_restormer_film \
    --dataset rdd --epochs 200 --res 256 --model_variant v2 \
    --output ./experiment_results/restormer_film_rdd/
```

CrossAttn is trained at 192x192 (`--res 192`) because its key/value projections do
not fit in 24 GB at 256x256. All other runs use 256x256.

---

## Evaluation

```bash
python evaluate_checkpoint.py --model shadow_guided_restormer_concat \
    --ckpt experiment_results/restormer_concat_sd7k/shadow_guided_restormer_concat_best.pth \
    --dataset sd7k

python eval_efficiency.py     # params, FLOPs, time, memory
python eval_lpips.py          # perceptual metric
python eval_ocr.py            # text-region preservation
python eval_cross_dataset.py  # zero-shot transfer between SD7K and RDD
```

---

## Results

PSNR (dB) and SSIM from the paper. Bold marks the best value in each column.

### Document shadow removal

| Strategy | SD7K (Restormer) | RDD (Restormer) | SD7K (NAFNet) | RDD (NAFNet) |
|---|---|---|---|---|
| No fusion | 23.43 / 0.9174 | 35.96 / 0.9753 | 21.94 / - | 33.52 / - |
| Concat | 24.40 / 0.9274 | **37.30** / 0.9738 | 22.33 / - | 34.27 / - |
| CrossAttn | 23.76 / 0.9164 | 36.99 / 0.9701 | **22.96** / - | 34.93 / - |
| FiLM | **24.77** / **0.9308** | 37.01 / 0.9797 | 22.74 / - | 34.91 / - |
| Gated | 24.31 / 0.9243 | 36.97 / 0.9677 | 22.37 / - | 34.79 / - |
| ASF | 24.08 / 0.9295 | 34.66 / 0.9668 | 22.55 / - | 34.87 / - |
| Large | 24.51 / 0.9282 | 37.31 / 0.9749 | 22.85 / - | **35.38** / - |

### Low-light enhancement and deraining

| Strategy | LOL (Restormer) | Rain100H (Restormer) | LOL (NAFNet) | Rain100H (NAFNet) |
|---|---|---|---|---|
| No fusion | 21.47 / 0.8458 | 35.18 / 0.9433 | 20.36 / 0.7828 | 27.42 / 0.8240 |
| Concat | **23.45** / 0.8586 | 35.98 / 0.9504 | 20.90 / 0.7966 | 28.00 / 0.8476 |
| CrossAttn | 23.15 / 0.8562 | 35.83 / 0.9530 | **22.05** / 0.8144 | 27.50 / 0.8390 |
| FiLM | 22.77 / 0.8588 | 36.84 / 0.9540 | 20.79 / 0.7980 | 28.04 / 0.8484 |
| Gated | 22.60 / 0.8489 | 35.73 / 0.9484 | 21.33 / 0.8012 | 27.93 / 0.8464 |
| ASF | 22.63 / 0.8467 | 35.79 / 0.9489 | 20.65 / 0.7999 | 27.95 / 0.8473 |
| Large | 22.88 / 0.8569 | **37.62** / **0.9606** | 20.75 / 0.8121 | **30.26** / **0.8894** |

Notes that apply to both tables:

- CrossAttn is evaluated at 192x192 against 256x256 for the other configurations,
  so its PSNR is not directly comparable and its ranks are read as indicative.
- FiLM on RDD uses gradient clipping; without it, training diverged to NaN.
- ASF is the one mechanism designed in this study. The others are reference
  implementations of known mechanisms plus one capacity control.
- SSIM was not logged for the two NAFNet document-shadow runs; all rankings in the
  paper use PSNR.

---

## Reproducibility notes

- About 80 training runs: 56 main-grid runs (seven entries including the no-fusion
  baseline, two backbones, four datasets), 12 training-size subsets, and the
  stability, clipping and control runs reported in the paper.
- Each configuration was trained once. The reported differences do not capture
  variation across random seeds, so small gaps should be read with that in mind.
- Training logs, checkpoints and datasets are excluded from this repository by
  `.gitignore`; only source code and result summaries are tracked.
- The scaling study covers three mechanisms on SD7K at 30, 60, 100, 150 and 200
  training pairs (`run_scaling_curve.py`).

---

## License

MIT. See [LICENSE](./LICENSE).

---

## Citation

```bibtex
@article{unifuse2026,
  title   = {UniFuse: A Controlled Testbed and Practical Guidelines for
             Feature Fusion in Image Restoration},
  author  = {Wang, Zhaona and Chu, Yangyang and Mu, Lili and Dong, Jin},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Under review}
}
```
