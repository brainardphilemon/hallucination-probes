"""Turn (input, answer) text from ANY generator into the Qwen2.5-7B hidden states the detector expects.

Training features were made by replaying chat prompt + the generated answer through Qwen2.5-7B-Instruct and
keeping every layer's hidden state at the last answer token. The same replay works for answers written by any
other model or by a human: Qwen reads the text, so the output always has Qwen's shape (29, 3584). The detector
then judges how Qwen's internal state looks while reading that answer. This is an out-of-distribution use for
non-Qwen answers (training answers were Qwen's own greedy generations); check it on labelled data before trusting it.

    hprobes embed --records records.jsonl --output states.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hprobes.extract import (DEFAULT_MODEL, DEFAULT_MODEL_REVISION, DEFAULT_SYSTEM_PROMPT, chat_prompt,
                             compute_dtype, release_gpu, stack_last_token_features)
from hprobes.predict import pick, read_records


def embed_with(model, tokenizer, records: Sequence[dict[str, Any]], device: str, system_prompt: str,
               max_input_tokens: int = 512, max_answer_tokens: int = 32, log_every: int = 25) -> np.ndarray:
    """(N, layers + 1, hidden) float16 last-answer-token states for already loaded model and tokenizer."""
    import torch

    out = []
    with torch.inference_mode():
        for i, record in enumerate(records):
            question, answer = pick(record, "input"), pick(record, "generated")
            if question is None or answer is None:
                raise ValueError(f"Record {i} needs an input and a generated_output")
            prompt = chat_prompt(tokenizer, str(question), system_prompt)
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=max_input_tokens)["input_ids"][0]
            answer_ids = tokenizer(str(answer).strip(), add_special_tokens=False,
                                   return_tensors="pt")["input_ids"][0][:max_answer_tokens]
            ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(device)
            replay = model(ids, attention_mask=torch.ones_like(ids), output_hidden_states=True,
                           use_cache=False, return_dict=True)
            out.append(stack_last_token_features(replay.hidden_states, torch).numpy().astype(np.float16))
            if log_every and (i + 1) % log_every == 0:
                print(f"embed {i + 1}/{len(records)}", flush=True)
    return np.stack(out)


def embed_records(records: Sequence[dict[str, Any]], args) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = compute_dtype(torch, args.device)
    kwargs: dict[str, Any] = {"revision": args.model_revision, "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if not args.no_4bit:
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True), device_map={"": args.device})
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.no_4bit:
        model = model.to(args.device)
    model.eval()
    try:
        return embed_with(model, tokenizer, records, args.device, args.system_prompt,
                          args.max_input_tokens, args.max_answer_tokens)
    finally:
        del model, tokenizer
        release_gpu(torch)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True,
                        help="jsonl/json/csv with input and generated_output per example")
    parser.add_argument("--output", type=Path, required=True, help="states.npy to write, (N, 29, 3584) float16")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--no-4bit", action="store_true", help="Load Qwen without 4-bit quantization")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=32,
                        help="Training answers were at most 32 tokens; longer answers are cut")
    return parser


def run(args) -> np.ndarray:
    states = embed_records(read_records(args.records), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, states)
    print(f"Saved hidden states {states.shape} to {args.output}", flush=True)
    return states


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
