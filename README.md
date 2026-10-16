# hallucination-probes

Cross-domain hallucination detection from LLM hidden states. A detector is trained on the
last-answer-token hidden states of Qwen2.5-7B-Instruct (all 29 layers × 3584) from several QA
datasets and is evaluated on a **held-out dataset it never saw**. Labels are BLEURT-20 ≥ 0.5,
where 1 means correct (not hallucinated).

## Layout

```
src/hprobes/
  cli.py            hprobes extract | train | judge
  extract.py        answers + hidden states + BLEURT labels  -> <artifact-dir>/{generation,bleurt}/<domain>/
  domains.py        dataset loaders (tqa, triviaqa, sciq, nq_open, popqa, web_questions, hotpotqa) and presets
  data.py           labels, per-source stratified 67.5/7.5/25 train/val/test splits, lazy feature memmaps
  features.py       layer pooling (none / max / mean / logsumexp) and PCA (global or per-layer)
  losses.py         bce, focal, logit_adjusted
  training.py       shared training loop (best source-val AUROC checkpoint), residual MLP head
  reporting.py      threshold calibration on source-val, metrics, results.csv / summary.json
  methods/          sala, mlp, fusion, vlm, polysemantic
  sala_core/        ISR init + invariant layer projections (from the SALA paper code, MIT)
  judges/           Jev and LAYA LLM-judge baselines
notebooks/          kaggle_extract.ipynb, kaggle_train.ipynb
tests/              CPU smoke tests on synthetic features
```

## Install

```bash
pip install -e ".[extract,dev]"
```

## 1. Extract features (GPU)

```bash
hprobes extract --artifact-dir artifacts/qwen25 --domains tqa nq_open sciq popqa web_questions hotpotqa triviaqa
```

## 2. Train and evaluate

Every combination below is set by flags:

| Choice | Flag |
|---|---|
| Method | `--method sala` (per-layer invariant projections → MLP), `mlp` (SALA without projections: states → MLP), `fusion`, `vlm`, `polysemantic` |
| Datasets | `--sources tqa sciq ... --targets triviaqa [nq_open ...]` or `--preset G14_6SRC` |
| Layer pooling | `--pooling none \| max \| mean \| logsumexp` with `--pool-kernel 3 --pool-stride 2` (29 → 15 layers), or `--pool-layers 15` to always end with 15 layers whatever the model's layer count |
| PCA | `--pca-dim 512` (0 = off) with `--pca-scope per-layer \| global`; fitted on source-train only. `mlp` supports both scopes; `sala` supports per-layer only |
| Loss | `--loss bce \| focal \| logit_adjusted` (`--focal-alpha 0.25 --focal-gamma 2`) |
| MLP head | `--hidden-dim`, `--hidden-layers`, `--dropout` |
| Optimization | `--epochs --lr --lr-schedule constant\|cosine --batch-size --weight-decay --grad-clip --amp/--no-amp --seed` |
| SALA | `--proj-dim 32` or `--lodo-dim` (leave-one-domain-out choice from `--proj-dim-candidates`), `--lam-inv --lam-sep --lam-reg --epochs-proj` |
| Fusion | `--fusion-mode fused\|image\|text`, `--text-encoder`, `--finetune-text-encoder`, `--text-dropout` |
| VLM | `--patch-height --patch-width --vlm-text full\|none --vision-layers` |
| Polysemantic | `--poly-encoder perceiver\|full`, `--neuron-positions learned\|none`, `--no-descriptors`, `--latents` |

Defaults that you do not set depend on the method:

| Method | Epochs | lr / schedule | Head | AMP / clip |
|---|---|---|---|---|
| `sala` | 40 | 1e-3, constant | 1024 × 3 | off |
| `mlp` | 40 | 1e-3, constant | 128 × 4 | off |
| `fusion`, `vlm`, `polysemantic` | 10 | 3e-4, cosine | 128 × 4 | on / 1.0 |

Examples:

```bash
SRC="tqa nq_open sciq popqa web_questions hotpotqa"
# SALA, log-sum-exp pooling, LODO-selected projection dimensions
hprobes train --method sala --artifact-dir artifacts/qwen25 --output-dir runs/sala_lse \
  --sources $SRC --targets triviaqa --pooling logsumexp --lodo-dim
# SALA without projections: max pooling -> per-layer PCA 512 -> MLP, focal loss
hprobes train --method mlp --artifact-dir artifacts/qwen25 --output-dir runs/mlp_max_pca_focal \
  --sources $SRC --targets triviaqa --pooling max --pca-dim 512 --loss focal
# Raw states (no pooling, no PCA), BCE, two held-out targets
hprobes train --method mlp --artifact-dir artifacts/qwen25 --output-dir runs/mlp_raw \
  --sources tqa sciq popqa web_questions hotpotqa --targets triviaqa nq_open
```

Each run writes `summary.json` (configuration, source-val/test metrics, per-target AUROC and
balanced accuracy, method details) and `results.csv` (one row per target example). A run is
skipped if `summary.json` already exists, unless you pass `--overwrite`. SALA saves each layer's
projection under `sala_layers/`, so an interrupted run resumes where it stopped.

## 3. Score new examples

`hprobes train --method mlp` also writes `<output-dir>/model.pt` (pooling, fitted PCA, trained MLP and the
calibrated threshold). Feed it hidden states plus the texts:

```bash
hprobes predict --model runs/mlp/model.pt --states states.npy --records records.jsonl --output preds.csv
# add --bleurt to also score the generated output against the actual output and compare with the detector
```

* `states.npy`: last generated-answer-token hidden states of all 29 layers, shape `(N, 29, 3584)` or `(29, 3584)`,
  from Qwen2.5-7B-Instruct (same layers and chat prompt as `hprobes extract`).
* `records.jsonl` (optional, same order): `{"input": ..., "generated_output": ..., "actual_output": ...}`;
  `actual_output` may be a list of acceptable answers.
* The detector itself reads only the hidden states; the texts are reported and used for the BLEURT check.

```python
from hprobes.predict import Detector
out = Detector.load("runs/mlp/model.pt").predict(states)
out["probability_correct"], out["hallucinated"]   # label 1 = correct, 0 = hallucinated
```

### Models trained with the old SALA fork

`sala_raw_concat.py` runs (for example `nvthaai/sala-g14-logsumexp-layerpca512-4x128`) saved the MLP and
threshold but not the PCA. `hprobes import-legacy` rebuilds the PCA from the saved features with the same
seed, attaches the weights and checks the result against that run's `results.csv`:

```bash
hprobes import-legacy --run-dir <run folder with best_mlp.pt> --artifact-dir <features> --output model.pt
```

`notebooks/kaggle_inference.ipynb` does this and then scores your examples.

### PCA like pooling (`--pca-fit self`)

By default (`--pca-fit source`) the PCA is fitted once on source-train and its matrices are stored in `model.pt`, so
the detector only fits the LLM it was trained on. With `--pca-fit self` nothing is stored: every dataset, and at
inference the batch you pass to `hprobes predict`, is projected with components computed from its own rows (no
labels), per layer, signs fixed, ordered by variance, scaled by the layer's RMS spread. Combined with
`--pool-layers 15 --pca-dim 512`, any layer count and hidden size become 15 x 512. Needs a batch of at least 2
examples at inference (ideally more than the PCA dimension). Whether the MLP transfers across LLMs this way is untested.

### Several LLMs, one MLP

`hprobes train --method mlp --artifact-dir qwen=<dir> llama=<dir> --sources qwen:tqa llama:tqa ... --targets llama:triviaqa
--pooling logsumexp --pool-layers 15 --pca-dim 512` fits one PCA per LLM (15 x 512 numbers each) and trains one shared
MLP. `hprobes predict` picks the right PCA from the shape of `--states` (or `--llm`).

## LLM-judge baselines

```bash
TYPESAFE_API_KEY=... hprobes judge jev --artifact-dir artifacts/qwen25 --output-dir runs/jev --domain triviaqa
hprobes judge laya --artifact-dir artifacts/qwen25 --output-dir runs/laya --domain triviaqa
```

## Kaggle

See [docs/kaggle.md](docs/kaggle.md).

## Results so far

Qwen2.5-7B, 6 sources → TriviaQA held out, AUROC:

| Detector | TriviaQA AUROC |
|---|---|
| `mlp`, log-sum-exp pooling, 128 × 4 | 0.868 |
| `fusion`, frozen BGE text, 0% text dropout (3 seeds) | 0.869 ± 0.003 |
| `fusion`, image only (3 seeds) | 0.861 ± 0.006 |
| `vlm` from scratch, pooled | 0.861 |
| `vlm` from scratch, raw 29 × 3584 | 0.78–0.79 |

## Tests

```bash
pytest   # about 2 s on CPU, synthetic features, no downloads
```

## Credit

`src/hprobes/sala_core/` and the SALA method are adapted from the SALA paper's code (MIT License,
© The SALA Authors). The BLEURT-20 scorer is [bleurt-pytorch](https://github.com/lucadiliello/bleurt-pytorch).
