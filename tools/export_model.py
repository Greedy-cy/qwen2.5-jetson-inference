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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


MAGIC = b"QWENBIN1"
VERSION = 1
DATA_OFFSET = 1024 * 1024
FILE_HEADER = struct.Struct("<8sIIQQ")
ALIGNMENT = 256
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "int8": 1}


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


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def float32_to_bfloat16(array: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(array, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    result = ((bits + rounding_bias) >> 16).astype("<u2")
    nan_mask = np.isnan(values)
    if np.any(nan_mask):
        result[nan_mask] = ((bits[nan_mask] >> 16) | np.uint32(0x0040)).astype("<u2")
    return result


def bfloat16_to_float32(array: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(array, dtype="<u2").astype(np.uint32)
    return np.ascontiguousarray((words << 16).view(np.float32))


def bfloat16_error(reference: np.ndarray, storage: np.ndarray) -> tuple[float, float]:
    reconstructed = bfloat16_to_float32(storage).reshape(reference.shape)
    finite = np.isfinite(reference)
    if not np.any(finite):
        return 0.0, 0.0
    error = np.abs(reference[finite] - reconstructed[finite])
    return float(error.max(initial=0.0)), float(error.mean(dtype=np.float64))


def verify_archive(path: Path) -> dict:
    with path.open("rb") as source:
        mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(mapping) < FILE_HEADER.size:
                raise RuntimeError("archive is smaller than its fixed header")
            magic, version, json_size, data_offset, _ = FILE_HEADER.unpack_from(mapping, 0)
            if magic != MAGIC or version != VERSION:
                raise RuntimeError("archive magic or version mismatch")
            if FILE_HEADER.size + json_size > data_offset or data_offset >= len(mapping):
                raise RuntimeError("archive metadata bounds are invalid")
            metadata = json.loads(mapping[FILE_HEADER.size : FILE_HEADER.size + json_size])
            names: set[str] = set()
            dtype_counts = {name: 0 for name in DTYPE_BYTES}
            payload_bytes = 0
            for record in metadata["tensors"]:
                name = record["name"]
                if name in names:
                    raise RuntimeError(f"duplicate tensor {name}")
                names.add(name)
                dtype = record["dtype"]
                if dtype not in DTYPE_BYTES:
                    raise RuntimeError(f"unsupported dtype {dtype} for {name}")
                elements = 1
                for dimension in record["shape"]:
                    if dimension < 0:
                        raise RuntimeError(f"negative dimension for {name}")
                    elements *= dimension
                expected = elements * DTYPE_BYTES[dtype]
                if record["nbytes"] != expected:
                    raise RuntimeError(f"nbytes mismatch for {name}")
                if record["offset"] % ALIGNMENT != 0:
                    raise RuntimeError(f"unaligned tensor {name}")
                begin = data_offset + record["offset"]
                end = begin + record["nbytes"]
                if end > len(mapping):
                    raise RuntimeError(f"tensor outside archive: {name}")
                actual_sha256 = hashlib.sha256(mapping[begin:end]).hexdigest()
                if actual_sha256 != record["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch for {name}")
                dtype_counts[dtype] += 1
                payload_bytes += record["nbytes"]
            return {
                "metadata_precision": metadata["precision"],
                "tensor_count": len(metadata["tensors"]),
                "dtype_counts": dtype_counts,
                "payload_bytes": payload_bytes,
                "file_bytes": len(mapping),
                "all_tensor_sha256_verified": True,
            }
        finally:
            mapping.close()


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


def groupwise_reconstruction_error(
    weight: np.ndarray, quantized: np.ndarray, scales: np.ndarray, group_size: int
) -> tuple[float, float]:
    rows, columns = weight.shape
    groups = (columns + group_size - 1) // group_size
    if quantized.shape != weight.shape or scales.shape != (rows, groups):
        raise ValueError("groupwise reconstruction shapes do not match")
    max_error = 0.0
    error_sum = 0.0
    error_count = 0
    for group in range(groups):
        begin = group * group_size
        end = min(columns, begin + group_size)
        reconstructed = (
            quantized[:, begin:end].astype(np.float32) * scales[:, group, None]
        )
        error = np.abs(weight[:, begin:end] - reconstructed)
        max_error = max(max_error, float(error.max(initial=0.0)))
        error_sum += float(error.sum(dtype=np.float64))
        error_count += error.size
    return max_error, error_sum / max(error_count, 1)


def export(source: Path, destination: Path, precision: str, group_size: int) -> dict:
    if precision == "w8a16" and group_size != 64:
        raise ValueError("w8a16 currently requires group size 64")
    tensors, readers = discover_tensors(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    quantization: list[dict] = []
    bfloat16_conversion: list[dict] = []
    try:
        with destination.open("w+b") as output:
            output.truncate(DATA_OFFSET)
            output.seek(DATA_OFFSET)
            for index, tensor in enumerate(tensors, 1):
                array = readers[tensor.path].array(tensor)
                if precision == "w8a16" and should_quantize(tensor.name, array):
                    q, scales, _, _ = quantize_groupwise(array, group_size)
                    scale_storage = float32_to_bfloat16(scales)
                    scale_max_error, scale_mean_error = bfloat16_error(
                        scales, scale_storage
                    )
                    stored_scales = bfloat16_to_float32(scale_storage).reshape(
                        scales.shape
                    )
                    max_error, mean_error = groupwise_reconstruction_error(
                        array, q, stored_scales, group_size
                    )
                    scale_name = tensor.name + ".scale"
                    write_tensor(
                        output,
                        records,
                        tensor.name,
                        q,
                        "int8",
                        {
                            "scheme": "symmetric_group_w8a16",
                            "group_size": group_size,
                            "axis": 1,
                            "scale_tensor": scale_name,
                            "scale_dtype": "bfloat16",
                        },
                    )
                    write_tensor(
                        output, records, scale_name, scale_storage, "bfloat16"
                    )
                    quantization.append(
                        {
                            "name": tensor.name,
                            "scale_tensor": scale_name,
                            "scale_dtype": "bfloat16",
                            "groups_per_row": scales.shape[1],
                            "max_abs_error": max_error,
                            "mean_abs_error": mean_error,
                        }
                    )
                    bfloat16_conversion.append(
                        {
                            "name": scale_name,
                            "max_abs_error": scale_max_error,
                            "mean_abs_error": scale_mean_error,
                        }
                    )
                elif precision in ("w16a16", "w8a16"):
                    storage = float32_to_bfloat16(array)
                    max_error, mean_error = bfloat16_error(array, storage)
                    write_tensor(
                        output, records, tensor.name, storage, "bfloat16"
                    )
                    bfloat16_conversion.append(
                        {
                            "name": tensor.name,
                            "max_abs_error": max_error,
                            "mean_abs_error": mean_error,
                        }
                    )
                elif precision == "int8" and should_quantize(tensor.name, array):
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
                "group_size": group_size if precision in ("int8", "w8a16") else None,
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

    verification = verify_archive(destination)
    report = {
        "archive": str(destination),
        "archive_sha256": sha256_file(destination),
        "precision": precision,
        "bytes": destination.stat().st_size,
        "payload_bytes": verification["payload_bytes"],
        "tensor_count": len(records),
        "dtype_counts": verification["dtype_counts"],
        "verification": verification,
        "quantized_tensor_count": len(quantization),
        "max_groupwise_abs_error": max(
            (item["max_abs_error"] for item in quantization), default=0.0
        ),
        "mean_tensor_abs_error": float(
            np.mean([item["mean_abs_error"] for item in quantization])
        ) if quantization else 0.0,
        "quantized_tensors": quantization,
        "bfloat16_tensor_count": len(bfloat16_conversion),
        "max_bfloat16_abs_error": max(
            (item["max_abs_error"] for item in bfloat16_conversion), default=0.0
        ),
        "mean_tensor_bfloat16_abs_error": float(
            np.mean([item["mean_abs_error"] for item in bfloat16_conversion])
        ) if bfloat16_conversion else 0.0,
        "bfloat16_tensors": bfloat16_conversion,
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


def self_test() -> None:
    exact = np.array([0.0, -0.0, 1.0, -2.5, np.inf, -np.inf], dtype=np.float32)
    exact_roundtrip = bfloat16_to_float32(float32_to_bfloat16(exact))
    np.testing.assert_array_equal(exact_roundtrip, exact)

    tie_to_even = np.array([1.0 + 1.0 / 256.0], dtype=np.float32)
    assert int(float32_to_bfloat16(tie_to_even)[0]) == 0x3F80
    above_tie = np.nextafter(tie_to_even, np.float32(np.inf))
    assert int(float32_to_bfloat16(above_tie)[0]) == 0x3F81

    nan_roundtrip = bfloat16_to_float32(
        float32_to_bfloat16(np.array([np.nan], dtype=np.float32))
    )
    assert np.isnan(nan_roundtrip[0])
    print("BF16 exporter self-test passed")


def w8a16_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qwen-w8a16-export-") as temporary:
        source = Path(temporary) / "source"
        source.mkdir()
        tensors = {
            "model.embed_tokens.weight": np.linspace(
                -0.25, 0.25, 20, dtype=np.float32
            ).reshape(4, 5),
            "model.layers.0.self_attn.q_proj.weight": np.linspace(
                -0.75, 0.75, 210, dtype=np.float32
            ).reshape(3, 70),
            "model.layers.0.self_attn.q_proj.bias": np.array(
                [0.125, -0.25, 0.5], dtype=np.float32
            ),
            "model.norm.weight": np.linspace(0.5, 1.5, 5, dtype=np.float32),
            "lm_head.weight": np.linspace(
                -0.125, 0.125, 20, dtype=np.float32
            ).reshape(4, 5),
        }
        tensors["model.layers.0.self_attn.q_proj.weight"][2, :64] = 0.0
        header = {}
        payload = []
        offset = 0
        for name, array in tensors.items():
            storage = np.ascontiguousarray(array, dtype="<f4")
            header[name] = {
                "dtype": "F32",
                "shape": list(storage.shape),
                "data_offsets": [offset, offset + storage.nbytes],
            }
            payload.append(storage)
            offset += storage.nbytes
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with (source / "model.safetensors").open("wb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for array in payload:
                array.tofile(output)

        destination = Path(temporary) / "model.w8a16.qbin"
        report = export(source, destination, "w8a16", 64)
        assert report["precision"] == "w8a16"
        assert report["quantized_tensor_count"] == 1
        assert report["tensor_count"] == 6
        assert report["dtype_counts"]["int8"] == 1
        assert report["dtype_counts"]["bfloat16"] == 5
        assert report["max_groupwise_abs_error"] <= 0.007
        assert report["verification"]["all_tensor_sha256_verified"]

        with destination.open("rb") as archive:
            mapping = mmap.mmap(archive.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                _, _, json_size, data_offset, _ = FILE_HEADER.unpack_from(mapping, 0)
                metadata = json.loads(
                    mapping[FILE_HEADER.size : FILE_HEADER.size + json_size]
                )
                records = {item["name"]: item for item in metadata["tensors"]}
                weight_name = "model.layers.0.self_attn.q_proj.weight"
                scale_name = weight_name + ".scale"
                weight_record = records[weight_name]
                scale_record = records[scale_name]
                assert metadata["precision"] == "w8a16"
                assert metadata["group_size"] == 64
                assert weight_record["dtype"] == "int8"
                assert weight_record["quant"]["scheme"] == "symmetric_group_w8a16"
                assert weight_record["quant"]["scale_dtype"] == "bfloat16"
                assert weight_record["quant"]["scale_tensor"] == scale_name
                assert scale_record["dtype"] == "bfloat16"
                assert scale_record["shape"] == [3, 2]
                assert records["model.embed_tokens.weight"]["dtype"] == "bfloat16"
                assert records["model.norm.weight"]["dtype"] == "bfloat16"
                assert records["lm_head.weight"]["dtype"] == "bfloat16"
                q = np.frombuffer(
                    mapping,
                    dtype=np.int8,
                    count=210,
                    offset=data_offset + weight_record["offset"],
                ).copy().reshape(3, 70)
                scale_storage = np.frombuffer(
                    mapping,
                    dtype="<u2",
                    count=6,
                    offset=data_offset + scale_record["offset"],
                ).copy().reshape(3, 2)
            finally:
                mapping.close()
        scales = bfloat16_to_float32(scale_storage).reshape(3, 2)
        max_error, _ = groupwise_reconstruction_error(
            tensors[weight_name], q, scales, 64
        )
        assert abs(max_error - report["max_groupwise_abs_error"]) < 1e-12
    print("W8A16 exporter self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-w8a16", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--precision",
        choices=("fp32", "int8", "w16a16", "w8a16", "all"),
        default="all",
    )
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.self_test_w8a16:
        w8a16_self_test()
        return
    if args.source is None or args.output is None:
        parser.error("--source and --output are required unless --self-test is used")
    if args.group_size <= 0:
        parser.error("group size must be positive")
    if args.precision in ("w8a16", "all") and args.group_size != 64:
        parser.error("w8a16 currently requires --group-size 64")

    copy_metadata(args.source, args.output)
    reports = []
    if args.precision in ("fp32", "all"):
        reports.append(export(args.source, args.output / "model.fp32.qbin", "fp32", args.group_size))
    if args.precision in ("int8", "all"):
        reports.append(export(args.source, args.output / "model.int8.qbin", "int8", args.group_size))
    if args.precision in ("w16a16", "all"):
        reports.append(
            export(
                args.source,
                args.output / "model.w16a16.qbin",
                "w16a16",
                args.group_size,
            )
        )
    if args.precision in ("w8a16", "all"):
        reports.append(
            export(
                args.source,
                args.output / "model.w8a16.qbin",
                "w8a16",
                args.group_size,
            )
        )
    report_path = args.output / "export_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
