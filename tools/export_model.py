#!/usr/bin/env python3
"""Export Hugging Face/ModelScope safetensors into mmap-friendly QWENBIN1 files.

The converter deliberately does not depend on PyTorch. BF16 is decoded by moving
its 16 payload bits into the high half of an IEEE FP32 value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


MAGIC = b"QWENBIN1"
VERSION = 1
DATA_OFFSET = 1024 * 1024
FILE_HEADER = struct.Struct("<8sIIQQ")
ALIGNMENT = 256


@dataclass(frozen=True)
class SourceTensor:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


class SafeTensorFile:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        header_size = struct.unpack_from("<Q", self.mapping, 0)[0]
        self.data_begin = 8 + header_size
        self.header = json.loads(self.mapping[8 : self.data_begin])

    def close(self) -> None:
        self.mapping.close()
        self.file.close()

    def array(self, tensor: SourceTensor) -> np.ndarray:
        raw = memoryview(self.mapping)[
            self.data_begin + tensor.begin : self.data_begin + tensor.end
        ]
        if tensor.dtype == "BF16":
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            return np.ascontiguousarray((words << 16).view(np.float32).reshape(tensor.shape))
        if tensor.dtype == "F16":
            return np.ascontiguousarray(
                np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(tensor.shape)
            )
        if tensor.dtype == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(tensor.shape).copy()
        raise ValueError(f"unsupported source dtype {tensor.dtype} for {tensor.name}")


def discover_tensors(source: Path) -> tuple[list[SourceTensor], dict[Path, SafeTensorFile]]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_paths = sorted({source / name for name in index["weight_map"].values()})
    else:
        shard_paths = sorted(source.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no safetensors files under {source}")

    files = {path: SafeTensorFile(path) for path in shard_paths}
    tensors: list[SourceTensor] = []
    for path, reader in files.items():
        for name, item in reader.header.items():
            if name == "__metadata__":
                continue
            begin, end = item["data_offsets"]
            tensors.append(
                SourceTensor(
                    name=name,
                    path=path,
                    dtype=item["dtype"],
                    shape=tuple(item["shape"]),
                    begin=begin,
                    end=end,
                )
            )
    tensors.sort(key=lambda item: item.name)
    return tensors, files


def align_file(output) -> int:
    relative = output.tell() - DATA_OFFSET
    padding = (-relative) % ALIGNMENT
    if padding:
        output.write(b"\0" * padding)
    return output.tell() - DATA_OFFSET


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def write_tensor(output, records: list[dict], name: str, array: np.ndarray,
                 dtype: str, quant: dict | None = None) -> None:
    array = np.ascontiguousarray(array)
    offset = align_file(output)
    array.tofile(output)
    record = {
        "name": name,
        "shape": list(array.shape),
        "dtype": dtype,
        "offset": offset,
        "nbytes": array.nbytes,
        "sha256": digest(array),
    }
    if quant is not None:
        record["quant"] = quant
    records.append(record)


def should_quantize(name: str, array: np.ndarray) -> bool:
    if array.ndim != 2:
        return False
    return name not in {"model.embed_tokens.weight", "lm_head.weight"}


def quantize_groupwise(weight: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    rows, columns = weight.shape
    groups = (columns + group_size - 1) // group_size
    quantized = np.empty_like(weight, dtype=np.int8)
    scales = np.empty((rows, groups), dtype=np.float32)
    max_error = 0.0
    error_sum = 0.0
    error_count = 0
    for group in range(groups):
        begin = group * group_size
        end = min(columns, begin + group_size)
        block = weight[:, begin:end]
        scale = np.max(np.abs(block), axis=1) / 127.0
        scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32)
        q = np.clip(np.rint(block / scale[:, None]), -127, 127).astype(np.int8)
        quantized[:, begin:end] = q
        scales[:, group] = scale
        error = np.abs(block - q.astype(np.float32) * scale[:, None])
        max_error = max(max_error, float(error.max(initial=0.0)))
        error_sum += float(error.sum(dtype=np.float64))
        error_count += error.size
    return quantized, scales, max_error, error_sum / max(error_count, 1)


def export(source: Path, destination: Path, precision: str, group_size: int) -> dict:
    tensors, readers = discover_tensors(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    quantization: list[dict] = []
    try:
        with destination.open("w+b") as output:
            output.truncate(DATA_OFFSET)
            output.seek(DATA_OFFSET)
            for index, tensor in enumerate(tensors, 1):
                array = readers[tensor.path].array(tensor)
                if precision == "int8" and should_quantize(tensor.name, array):
                    q, scales, max_error, mean_error = quantize_groupwise(array, group_size)
                    scale_name = tensor.name + ".scale"
                    write_tensor(
                        output,
                        records,
                        tensor.name,
                        q,
                        "int8",
                        {
                            "scheme": "symmetric_group_w8a32",
                            "group_size": group_size,
                            "axis": 1,
                            "scale_tensor": scale_name,
                        },
                    )
                    write_tensor(output, records, scale_name, scales, "float32")
                    quantization.append(
                        {
                            "name": tensor.name,
                            "max_abs_error": max_error,
                            "mean_abs_error": mean_error,
                        }
                    )
                else:
                    write_tensor(output, records, tensor.name, array.astype(np.float32, copy=False), "float32")
                print(f"[{index:3d}/{len(tensors)}] {tensor.name} {tensor.shape}", flush=True)

            metadata = {
                "format": "QWENBIN1",
                "version": VERSION,
                "source": str(source),
                "precision": precision,
                "group_size": group_size if precision == "int8" else None,
                "tensors": records,
            }
            encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if FILE_HEADER.size + len(encoded) > DATA_OFFSET:
                raise RuntimeError("archive JSON exceeds reserved header region")
            output.seek(0)
            output.write(FILE_HEADER.pack(MAGIC, VERSION, len(encoded), DATA_OFFSET, 0))
            output.write(encoded)
    finally:
        for reader in readers.values():
            reader.close()

    report = {
        "archive": str(destination),
        "precision": precision,
        "bytes": destination.stat().st_size,
        "tensor_count": len(records),
        "quantized_tensor_count": len(quantization),
        "max_groupwise_abs_error": max(
            (item["max_abs_error"] for item in quantization), default=0.0
        ),
        "mean_tensor_abs_error": float(
            np.mean([item["mean_abs_error"] for item in quantization])
        ) if quantization else 0.0,
        "quantized_tensors": quantization,
    }
    return report


def copy_metadata(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, output / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "int8", "all"), default="all")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error("group size must be positive")

    copy_metadata(args.source, args.output)
    reports = []
    if args.precision in ("fp32", "all"):
        reports.append(export(args.source, args.output / "model.fp32.qbin", "fp32", args.group_size))
    if args.precision in ("int8", "all"):
        reports.append(export(args.source, args.output / "model.int8.qbin", "int8", args.group_size))
    report_path = args.output / "export_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
