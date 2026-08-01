#!/usr/bin/env python3
"""Reproducible end-to-end parity check for the C++ FP32/INT8 runtimes."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cpp_tokens(command: list[str]) -> list[int]:
    output = subprocess.check_output(command, text=True)
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return [int(item) for item in value]
    raise RuntimeError("C++ output did not contain token-id JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/llm_infer"))
    parser.add_argument("--prompt", default="用一句话解释什么是大语言模型。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cpp_results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as temporary:
        for precision in ("fp32", "int8"):
            logits_path = Path(temporary) / f"{precision}.f32"
            common = [
                "--model", str(args.runtime), "--backend", "cuda",
                "--precision", precision, "--max-seq-len", "256", "--prompt", args.prompt,
            ]
            subprocess.run(
                [str(args.binary), "logits", *common, "--output", str(logits_path)],
                check=True,
            )
            tokens = cpp_tokens(
                [str(args.binary), "generate", *common, "--max-new-tokens",
                 str(args.max_new_tokens), "--show-token-ids"]
            )
            cpp_results[precision] = {
                "logits": np.fromfile(logits_path, dtype="<f4"),
                "tokens": tokens,
            }

    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.float32, local_files_only=True).eval()
    with torch.inference_mode():
        reference_logits = model(**inputs).logits[0, -1].float().numpy().copy()
        reference_tokens = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
            use_cache=True)[0, inputs.input_ids.shape[1]:].tolist()

    reports = []
    for precision, cpp in cpp_results.items():
        values = cpp["logits"]
        reports.append({
            "precision": precision,
            "top1_reference": int(reference_logits.argmax()),
            "top1_cpp": int(values.argmax()),
            "max_abs": float(np.max(np.abs(reference_logits - values))),
            "mean_abs": float(np.mean(np.abs(reference_logits - values))),
            "cosine": float(np.dot(reference_logits, values) /
                            (np.linalg.norm(reference_logits) * np.linalg.norm(values))),
            "reference_tokens": reference_tokens,
            "cpp_tokens": cpp["tokens"],
            "token_match": reference_tokens == cpp["tokens"],
        })

    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
