"""QA datasets supported for answer generation, and named source/target presets."""

from __future__ import annotations

import json
import random
from typing import Any, Iterable

DATASET_SPECS = {
    "tqa": {
        "repo": "truthfulqa/truthful_qa",
        "revision": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        "config": "generation",
        "split": "validation",
    },
    "triviaqa": {
        "repo": "mandarjoshi/trivia_qa",
        "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        "config": "rc.nocontext",
        "split": "validation",
    },
    "sciq": {
        "repo": "allenai/sciq",
        "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
        "config": None,
        "split": "validation",
    },
    "nq_open": {
        "repo": "google-research-datasets/nq_open",
        "revision": "5dd9790a83002ad084ddeb7c420dc716852c6f28",
        "config": "nq_open",
        "split": "validation",
    },
    "popqa": {
        "repo": "akariasai/PopQA",
        "revision": "098765c79ea10a2cb19c828324e33281b8336ec0",
        "config": None,
        "split": "test",
    },
    "web_questions": {
        "repo": "stanfordnlp/web_questions",
        "revision": "0e473cbe21d1e91ec18da343644498be6a3f5454",
        "config": None,
        "split": "train",
    },
    "hotpotqa": {
        "repo": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "config": "distractor",
        "split": "validation",
    },
}

DOMAINS = tuple(DATASET_SPECS)

# Named (sources, target) presets kept from the SALA experiments. Any combination of
# DOMAINS can be passed directly with --sources / --targets instead.
PRESETS = {
    "G3": {"sources": ["tqa"], "targets": ["triviaqa"]},
    "G5": {"sources": ["sciq"], "targets": ["nq_open"]},
    "G14": {"sources": ["tqa", "nq_open", "sciq"], "targets": ["triviaqa"]},
    "G15": {"sources": ["tqa", "sciq", "triviaqa"], "targets": ["nq_open"]},
    "G14_6SRC": {"sources": ["tqa", "nq_open", "sciq", "popqa", "web_questions", "hotpotqa"],
                 "targets": ["triviaqa"]},
    "G15_6SRC": {"sources": ["tqa", "sciq", "popqa", "web_questions", "hotpotqa", "triviaqa"],
                 "targets": ["nq_open"]},
}


def _unique_texts(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _json_string_list(value: Any, domain: str, field: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{domain} field {field} is not a valid JSON list") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{domain} field {field} must be a list")
    return _unique_texts(value)


def load_domain_records(domain: str, args) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASET_SPECS[domain]
    kwargs = {
        "path": spec["repo"],
        "split": spec["split"],
        "revision": spec["revision"],
        "cache_dir": args.cache_dir,
    }
    if spec["config"] is not None:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(**kwargs)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, example in enumerate(dataset):
        if domain == "tqa":
            example_id = str(index)
            references = _unique_texts([example["best_answer"], *example["correct_answers"]])
        elif domain == "triviaqa":
            example_id = str(example["question_id"])
            if example_id in seen:
                continue
            references = _unique_texts(example["answer"]["aliases"])
        elif domain == "sciq":
            example_id = str(index)
            references = _unique_texts([example["correct_answer"]])
        elif domain == "nq_open":
            example_id = str(index)
            references = _unique_texts(example["answer"])
        elif domain == "popqa":
            example_id = str(example["id"])
            references = _json_string_list(example["possible_answers"], domain, "possible_answers")
        elif domain == "web_questions":
            example_id = str(index)
            references = _unique_texts(example["answers"])
        elif domain == "hotpotqa":
            example_id = str(example["id"])
            references = _unique_texts([example["answer"]])
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        seen.add(example_id)
        if not references:
            raise ValueError(f"{domain}:{example_id} has no reference answers")
        records.append({
            "id": example_id,
            "input_text": str(example["question"]).strip(),
            "ground_truth_answers": references,
        })
    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    if args.samples_per_domain:
        order = order[: min(args.samples_per_domain, len(order))]
    return [records[index] for index in order]
