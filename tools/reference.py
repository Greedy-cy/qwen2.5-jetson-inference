#!/usr/bin/env python3
"""Generate tokenizer, logits and greedy-token golden data with Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CASES = [
    "Hello, introduce yourself in one sentence.",
    "用一句话解释什么是大语言模型。",
    "CUDA + C++ = 🚀\n保留空格  和换行。",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True
    ).to(args.device).eval()
    manifest = []
    for index, prompt in enumerate(CASES):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float().cpu().numpy()
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
            )[0, inputs.input_ids.shape[1] :].cpu().tolist()
        logits_path = args.output / f"case_{index}_last_logits.f32"
        np.asarray(logits, dtype="<f4").tofile(logits_path)
        manifest.append(
            {
                "prompt": prompt,
                "chat_text": text,
                "input_ids": inputs.input_ids[0].cpu().tolist(),
                "generated_ids": generated,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
                "last_logits": logits_path.name,
            }
        )
    (args.output / "reference.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
