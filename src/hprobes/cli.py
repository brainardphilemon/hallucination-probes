"""Command line: ``hprobes extract | train | predict | generate | import-legacy | judge``.

Examples::

    # 1. answers + hidden states + BLEURT labels for the datasets you need
    hprobes extract --artifact-dir art --domains tqa nq_open sciq popqa web_questions hotpotqa triviaqa

    # 2. SALA (projections) on six sources, TriviaQA held out
    hprobes train --method sala --artifact-dir art --output-dir runs/sala --preset G14_6SRC --lodo-dim

    # SALA without projections: log-sum-exp pooled states -> per-layer PCA 512 -> 4x128 MLP, focal loss
    hprobes train --method mlp --artifact-dir art --output-dir runs/mlp \\
        --sources tqa nq_open sciq popqa web_questions hotpotqa --targets triviaqa \\
        --pooling logsumexp --pca-dim 512 --pca-scope per-layer --loss focal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from hprobes.domains import DOMAINS, PRESETS
from hprobes.features import POOLING
from hprobes.losses import LOSSES

METHODS = ("sala", "mlp", "fusion", "vlm", "polysemantic")

# Per-method defaults for options left unset on the command line.
METHOD_DEFAULTS = {
    "sala": dict(epochs=40, lr=1e-3, lr_schedule="constant", hidden_dim=1024, hidden_layers=3, amp=False,
                 grad_clip=0.0, batch_size=128),
    "mlp": dict(epochs=40, lr=1e-3, lr_schedule="constant", hidden_dim=128, hidden_layers=4, amp=False,
                grad_clip=0.0, batch_size=128),
    "fusion": dict(epochs=10, lr=3e-4, lr_schedule="cosine", hidden_dim=128, hidden_layers=4, amp=True,
                   grad_clip=1.0, batch_size=128),
    "vlm": dict(epochs=10, lr=3e-4, lr_schedule="cosine", hidden_dim=128, hidden_layers=4, amp=True,
                grad_clip=1.0, batch_size=128),
    "polysemantic": dict(epochs=10, lr=3e-4, lr_schedule="cosine", hidden_dim=128, hidden_layers=4, amp=True,
                         grad_clip=1.0, batch_size=128),
}


def add_train_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--method", choices=METHODS, required=True)
    g.add_argument("--artifact-dir", nargs="+", required=True,
                   help="Output of `hprobes extract`; for several LLMs give <llm>=<dir> pairs and name datasets "
                        "<llm>:<dataset> (method mlp with --pca-dim)")
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--sources", nargs="+", help=f"Training datasets: {', '.join(DOMAINS)}")
    g.add_argument("--targets", nargs="+", help="Held-out test datasets (never trained on)")
    g.add_argument("--preset", choices=sorted(PRESETS), help="Named sources/targets (overridden by --sources/--targets)")
    g.add_argument("--bleurt-threshold", type=float, default=0.5, help="BLEURT >= threshold counts as correct")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--num-workers", type=int, default=2)
    g.add_argument("--overwrite", action="store_true", help="Re-run even if summary.json already exists")
    g.add_argument("--no-save-model", action="store_true",
                   help="Do not write <output-dir>/model.pt (method mlp only; used by `hprobes predict`)")

    g = p.add_argument_group("features")
    g.add_argument("--pooling", choices=POOLING, default="none", help="Pool hidden states across layers")
    g.add_argument("--pool-kernel", type=int, default=3, help="Layers per pooling window")
    g.add_argument("--pool-stride", type=int, default=2)
    g.add_argument("--pool-layers", type=int, default=0,
                   help="Pool any number of layers down to exactly this many (e.g. 15); 0 = use kernel/stride")
    g.add_argument("--pca-dim", type=int, default=0, help="PCA components fitted on source-train (0 = off; mlp, sala)")
    g.add_argument("--pca-fit", choices=("source", "self", "avg", "dct"), default="source",
                   help="how --pca-dim is reached. source: PCA fitted once on source-train and stored (same LLM at "
                        "inference); self: PCA computed from the rows it is applied to (any LLM, needs a batch); "
                        "avg / dct: data-free per-example reductions (mean of neighbouring hidden units / first "
                        "DCT coefficients), any LLM, even a single example")
    g.add_argument("--standardize", action="store_true",
                   help="method mlp: z-score every MLP input feature with the mean / std of each LLM's source-train "
                        "rows (saved in model.pt; at inference a new LLM can use its own statistics)")
    g.add_argument("--reduce-norm", choices=("rms", "none"), default="rms",
                   help="avg / dct only: divide each layer's reduced vector by its RMS")
    g.add_argument("--pca-scope", choices=("global", "per-layer"), default="per-layer",
                   help="One PCA over all features or one per (pooled) layer")
    g.add_argument("--feature-clip", type=float, default=10.0, help="Clamp standardized states (transformer methods)")

    g = p.add_argument_group("loss and optimization (unset values use per-method defaults)")
    g.add_argument("--loss", choices=LOSSES, default="bce")
    g.add_argument("--focal-alpha", type=float, default=0.25)
    g.add_argument("--focal-gamma", type=float, default=2.0)
    g.add_argument("--logit-adjust-tau", type=float, default=1.0)
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--lr-schedule", choices=("constant", "cosine"))
    g.add_argument("--warmup-ratio", type=float, default=0.05)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--grad-clip", type=float)
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Mixed precision on CUDA")
    g.add_argument("--hidden-dim", type=int, help="MLP head width")
    g.add_argument("--hidden-layers", type=int, help="MLP head hidden layers")
    g.add_argument("--dropout", type=float, default=0.2)

    g = p.add_argument_group("sala")
    g.add_argument("--proj-dim", type=int, default=32, help="Fixed per-layer projection dimension")
    g.add_argument("--lodo-dim", action="store_true", help="Choose each layer's dimension by leave-one-domain-out")
    g.add_argument("--proj-dim-candidates", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    g.add_argument("--epochs-proj", type=int, default=40)
    g.add_argument("--epochs-proj-selection", type=int, default=40)
    g.add_argument("--epochs-probe-selection", type=int, default=40)
    g.add_argument("--lam-inv", type=float, default=1.0)
    g.add_argument("--lam-sep", type=float, default=0.4)
    g.add_argument("--lam-reg", type=float, default=0.1)

    g = p.add_argument_group("transformer methods (fusion, vlm, polysemantic)")
    g.add_argument("--model-dim", type=int, default=256)
    g.add_argument("--heads", type=int, default=4)
    g.add_argument("--fusion-layers", type=int, default=2)

    g = p.add_argument_group("fusion")
    g.add_argument("--fusion-mode", choices=("fused", "image", "text"), default="fused")
    g.add_argument("--text-encoder", default="BAAI/bge-base-en-v1.5")
    g.add_argument("--text-encoder-revision", default="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
    g.add_argument("--finetune-text-encoder", action="store_true")
    g.add_argument("--text-encoder-lr", type=float, default=2e-5)
    g.add_argument("--text-dropout", type=float, default=0.0, help="Fraction of training samples without text")
    g.add_argument("--max-text-tokens", type=int, default=64)

    g = p.add_argument_group("vlm")
    g.add_argument("--vlm-text", choices=("full", "none"), default="full")
    g.add_argument("--patch-height", type=int, default=1, help="Layers per image patch")
    g.add_argument("--patch-width", type=int, default=256, help="Hidden units per image patch")
    g.add_argument("--vision-layers", type=int, default=2)
    g.add_argument("--max-question-tokens", type=int, default=96)
    g.add_argument("--max-answer-tokens", type=int, default=48)

    g = p.add_argument_group("polysemantic")
    g.add_argument("--poly-encoder", choices=("perceiver", "full"), default="perceiver")
    g.add_argument("--neuron-positions", choices=("learned", "none"), default="none")
    g.add_argument("--no-descriptors", dest="descriptors", action="store_false")
    g.add_argument("--latents", type=int, default=64)
    g.add_argument("--latent-heads", type=int, default=8)
    g.add_argument("--embedding-dim", type=int, default=512)


def resolve_train_args(args) -> argparse.Namespace:
    for key, value in METHOD_DEFAULTS[args.method].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.preset:
        args.sources = args.sources or PRESETS[args.preset]["sources"]
        args.targets = args.targets or PRESETS[args.preset]["targets"]
    if not args.sources or not args.targets:
        raise SystemExit("Give --sources and --targets, or --preset")
    if args.pca_dim < 0 or args.epochs < 1:
        raise SystemExit("--pca-dim must be >= 0 and --epochs >= 1")
    return args


def parse_artifacts(values):
    """One folder, or <llm>=<folder> pairs for several LLMs."""
    if len(values) == 1 and "=" not in values[0]:
        return Path(values[0])
    if any("=" not in v for v in values):
        raise SystemExit("With several --artifact-dir values write each as <llm>=<folder>")
    return {name: Path(path) for name, path in (v.split("=", 1) for v in values)}


def train(args) -> dict:
    import importlib

    from hprobes.data import load_experiment
    from hprobes.reporting import save_results
    from hprobes.training import get_device

    args = resolve_train_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    args.output_dir = output_dir
    if (output_dir / "summary.json").exists() and not args.overwrite:
        print(f"{output_dir / 'summary.json'} exists; pass --overwrite to re-run", flush=True)
        return json.loads((output_dir / "summary.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_experiment(parse_artifacts(args.artifact_dir), args.sources, args.targets, args.bleurt_threshold)
    if len(data.models) > 1 and args.method != "mlp":
        raise SystemExit("Training on several LLMs is implemented for --method mlp")
    device = get_device(args.device)
    print(f"{args.method}: sources={args.sources} targets={args.targets} "
          f"LLM shapes={dict(data.models)} device={device}", flush=True)
    method = importlib.import_module(f"hprobes.methods.{args.method}")
    method_summary, val_logits, test_logits, target_logits = method.run(args, data, device)
    summary = save_results(args, data, method_summary, val_logits, test_logits, target_logits)
    exported = getattr(method, "LAST_MODEL", None)
    if exported and not args.no_save_model:
        from hprobes.predict import save_bundle

        save_bundle(output_dir / "model.pt", exported, summary)
        print(f"Saved inference bundle to {output_dir / 'model.pt'}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    from hprobes import extract

    parser = argparse.ArgumentParser(prog="hprobes", description="Hallucination probes on LLM hidden states")
    sub = parser.add_subparsers(dest="command", required=True)
    extract.build_parser(sub.add_parser("extract", help="Generate answers, hidden states and BLEURT labels"))
    add_train_args(sub.add_parser("train", help="Train and evaluate a detector",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter))
    from hprobes import predict as predict_module

    predict_module.build_parser(sub.add_parser("predict", help="Score new examples with a trained model.pt"))
    from hprobes import generate as generate_module
    from hprobes import legacy

    legacy.build_parser(sub.add_parser("import-legacy", help="Turn an old sala_raw_concat.py run into a model.pt"))
    generate_module.build_parser(sub.add_parser("generate", help="Answers + hidden states from any chat model"))
    judge = sub.add_parser("judge", help="LLM-judge baselines over saved answers (Jev, LAYA)")
    judge.add_argument("judge", choices=("jev", "laya"))
    judge.add_argument("judge_args", nargs=argparse.REMAINDER, help="Arguments passed to the judge script")
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract.run(args)
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        predict_module.run(args)
    elif args.command == "generate":
        generate_module.run(args)
    elif args.command == "import-legacy":
        legacy.rebuild(args.run_dir, args.artifact_dir, args.output, args.device)
    else:
        from hprobes.judges import jev, laya

        module = jev if args.judge == "jev" else laya
        sys.argv = [f"hprobes judge {args.judge}", *args.judge_args]
        module.main()


if __name__ == "__main__":
    main()
