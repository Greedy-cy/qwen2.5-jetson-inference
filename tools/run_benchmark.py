#!/usr/bin/env python3
"""Reproducible benchmark runner for the Jetson Qwen inference runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CANONICAL_TOKEN_IDS = [
    151644, 8948, 198, 2610, 525, 264, 10950, 17847,
    13, 151645, 198, 151644, 872, 198, 840, 20772,
    12, 80285, 12, 17269, 12, 7053, 3098, 12,
    4086, 44, 13, 151645, 198, 151644, 77091, 198,
]
ARCHIVE_NAMES = {
    "fp32": "model.fp32.qbin",
    "int8": "model.int8.qbin",
    "w8a32": "model.int8.qbin",
    "w16a16": "model.w16a16.qbin",
    "w8a16": "model.w8a16.qbin",
}
BEGIN_MARKER = "BENCHMARK_MEASURE_BEGIN"
END_MARKER = "BENCHMARK_MEASURE_END"


def run_text(command: list[str], cwd: Path | None = None, check: bool = False) -> str:
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=check,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if check:
            raise RuntimeError(f"command failed: {' '.join(command)}") from error
        return f"unavailable: {error}"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values)
    return {
        "mean": mean,
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "stddev": stddev,
        "cv_percent": stddev * 100.0 / abs(mean) if mean else 0.0,
    }


def parse_tegrastats_line(line: str, timestamp: float) -> dict[str, Any]:
    sample: dict[str, Any] = {"timestamp_monotonic": timestamp}
    ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if ram:
        sample["ram_used_mb"] = int(ram.group(1))
        sample["ram_total_mb"] = int(ram.group(2))
    power = re.search(r"VDD_IN\s+(\d+)mW(?:/(\d+)mW)?", line)
    if power:
        sample["vdd_in_mw"] = int(power.group(1))
    gpu = re.search(r"GR3D_FREQ\s+(\d+)%@(?:\[(\d+)(?:,\s*\d+)*\]|(\d+))", line)
    if gpu:
        sample["gpu_util_percent"] = int(gpu.group(1))
        sample["gpu_frequency_mhz"] = int(gpu.group(2) or gpu.group(3))
    temperatures = {
        name: float(value)
        for name, value in re.findall(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C", line)
    }
    if temperatures:
        sample["temperatures_c"] = temperatures
    return sample


class TelemetrySampler:
    def __init__(self, interval_ms: int, enabled: bool) -> None:
        self.interval_ms = interval_ms
        self.enabled = enabled
        self.samples: list[dict[str, Any]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self.process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            self.error = "tegrastats not found"
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.samples.append(parse_tegrastats_line(line, time.monotonic()))

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)


class ClockGuard:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.state_file = Path(f"/tmp/qwen-benchmark-clocks-{os.getpid()}.conf")
        self.stored = False

    def __enter__(self) -> "ClockGuard":
        if not self.enabled:
            return self
        run_text(["sudo", "-n", "jetson_clocks", "--store", str(self.state_file)], check=True)
        self.stored = True
        try:
            run_text(["sudo", "-n", "jetson_clocks"], check=True)
        except RuntimeError:
            try:
                run_text(["sudo", "-n", "jetson_clocks", "--restore", str(self.state_file)], check=True)
            except RuntimeError as restore_error:
                print(f"warning: failed to restore clocks: {restore_error}", file=sys.stderr)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.enabled and self.stored:
            try:
                run_text(
                    ["sudo", "-n", "jetson_clocks", "--restore", str(self.state_file)],
                    check=True,
                )
            except RuntimeError as error:
                print(f"warning: failed to restore clocks: {error}", file=sys.stderr)


def run_benchmark_process(command: list[str]) -> tuple[dict[str, Any], dict[str, float], str]:
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    markers: dict[str, float] = {}
    diagnostic_lines: list[str] = []

    def read_stderr() -> None:
        assert process.stderr is not None
        for raw_line in process.stderr:
            line = raw_line.strip()
            now = time.monotonic()
            if line == BEGIN_MARKER:
                markers["begin"] = now
            elif line == END_MARKER:
                markers["end"] = now
            elif line:
                diagnostic_lines.append(line)

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    assert process.stdout is not None
    stdout = process.stdout.read()
    return_code = process.wait()
    stderr_thread.join(timeout=5)
    diagnostics = "\n".join(diagnostic_lines)
    if return_code != 0:
        raise RuntimeError(
            f"benchmark exited with {return_code}: {diagnostics or stdout.strip()}"
        )
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"benchmark did not emit valid JSON: {stdout[-1000:]}") from error
    return report, markers, diagnostics


def integrate_energy(samples: list[dict[str, Any]]) -> float:
    points = [
        (sample["timestamp_monotonic"], sample["vdd_in_mw"] / 1000.0)
        for sample in samples if "vdd_in_mw" in sample
    ]
    return sum(
        (left[1] + right[1]) * 0.5 * (right[0] - left[0])
        for left, right in zip(points, points[1:])
    )


def summarize_telemetry(
    sampler: TelemetrySampler, markers: dict[str, float], output_tokens: int,
) -> dict[str, Any]:
    if sampler.error:
        return {"available": False, "error": sampler.error}
    begin = markers.get("begin")
    end = markers.get("end")
    if begin is None or end is None or end <= begin:
        return {"available": False, "error": "measurement markers were not observed"}
    samples = [s for s in sampler.samples if begin <= s["timestamp_monotonic"] <= end]
    result: dict[str, Any] = {
        "available": bool(samples),
        "sample_count": len(samples),
        "measurement_window_seconds": end - begin,
    }
    if not samples:
        result["error"] = "no tegrastats samples inside measurement window"
        return result
    for key, output_key, scale in (
        ("vdd_in_mw", "vdd_in_w", 0.001),
        ("ram_used_mb", "ram_used_mb", 1.0),
        ("gpu_util_percent", "gpu_util_percent", 1.0),
        ("gpu_frequency_mhz", "gpu_frequency_mhz", 1.0),
    ):
        values = [float(sample[key]) * scale for sample in samples if key in sample]
        if values:
            result[output_key] = summarize_values(values)
    temperature_max: dict[str, float] = {}
    for sample in samples:
        for name, value in sample.get("temperatures_c", {}).items():
            temperature_max[name] = max(value, temperature_max.get(name, value))
    result["temperature_max_c"] = temperature_max
    energy = integrate_energy(samples)
    result["measurement_energy_joules"] = energy
    result["energy_per_generated_token_joules"] = (
        energy / output_tokens if output_tokens > 0 else None
    )
    result["samples"] = [
        {
            "timestamp_seconds": sample["timestamp_monotonic"] - begin,
            **{key: value for key, value in sample.items() if key != "timestamp_monotonic"},
        }
        for sample in samples
    ]
    return result


def validate_benchmark(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 2:
        raise RuntimeError("benchmark schema_version must be 2")
    for key in ("configuration", "tokens", "statistics", "samples", "memory"):
        if key not in report:
            raise RuntimeError(f"benchmark report lacks {key}")
    repeat = report["configuration"]["repeat"]
    if len(report["samples"]) != repeat:
        raise RuntimeError("sample count does not match repeat")
    generated = report["tokens"]["generated_per_iteration"]
    if generated <= 0 or any(s["generated_tokens"] != generated for s in report["samples"]):
        raise RuntimeError("generated token count is not deterministic")


def archive_for(model: Path, precision: str) -> Path:
    try:
        return model / ARCHIVE_NAMES[precision]
    except KeyError as error:
        raise RuntimeError(f"no archive mapping for precision {precision}") from error


def collect_provenance(root: Path, model: Path, precision: str) -> dict[str, Any]:
    status = run_text(["git", "status", "--porcelain"], cwd=root, check=True)
    archive = archive_for(model, precision)
    if not archive.is_file():
        raise RuntimeError(f"model archive not found: {archive}")
    default_nvcc = Path("/usr/local/cuda/bin/nvcc")
    nvcc = str(default_nvcc) if default_nvcc.is_file() else "nvcc"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": run_text(["git", "rev-parse", "HEAD"], cwd=root, check=True),
        "git_dirty": bool(status),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model_directory": str(model.resolve()),
        "model_archive": str(archive.resolve()),
        "model_archive_sha256": sha256_file(archive),
        "canonical_token_ids_sha256": hashlib.sha256(
            json.dumps(CANONICAL_TOKEN_IDS, separators=(",", ":")).encode()
        ).hexdigest(),
        "nvidia_tegra_release": Path("/etc/nv_tegra_release").read_text(
            encoding="utf-8", errors="replace"
        ).strip() if Path("/etc/nv_tegra_release").exists() else "unavailable",
        "cuda_compiler": run_text([nvcc, "--version"]),
        "power_mode": run_text(["nvpmodel", "-q"]),
        "cmake": run_text(["cmake", "--version"]).splitlines()[0],
    }


def self_test() -> None:
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == 2.5 and summary["p05"] == 1.0 and summary["p95"] == 4.0
    sample = parse_tegrastats_line(
        "RAM 2048/7620MB GR3D_FREQ 75%@[612] VDD_IN 5000mW/4800mW cpu@45.5C", 1.0
    )
    assert sample["ram_used_mb"] == 2048
    assert sample["gpu_frequency_mhz"] == 612
    assert sample["vdd_in_mw"] == 5000
    assert sample["temperatures_c"]["cpu"] == 45.5
    synthetic = {
        "schema_version": 2,
        "configuration": {"repeat": 1},
        "tokens": {"generated_per_iteration": 2},
        "statistics": {},
        "samples": [{"generated_tokens": 2}],
        "memory": {},
    }
    validate_benchmark(synthetic)
    assert len(CANONICAL_TOKEN_IDS) == 32
    print("benchmark runner self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--binary", type=Path, default=Path("build/llm_infer"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=tuple(ARCHIVE_NAMES), default="fp32")
    parser.add_argument("--token-ids", default=",".join(map(str, CANONICAL_TOKEN_IDS)))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--telemetry-interval-ms", type=int, default=100)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--lock-clocks", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--cublas", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.model is None or args.output is None:
        raise RuntimeError("--model and --output are required")
    if args.telemetry_interval_ms <= 0:
        raise RuntimeError("--telemetry-interval-ms must be positive")
    root = args.project_root.resolve()
    binary = (root / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    model = (root / args.model).resolve() if not args.model.is_absolute() else args.model
    provenance = collect_provenance(root, model, args.precision)
    if provenance["git_dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to benchmark a dirty worktree; use --allow-dirty for smoke tests")
    runtime_precision = "int8" if args.precision == "w8a32" else args.precision
    command = [
        str(binary), "benchmark", "--model", str(model),
        "--backend", args.backend, "--precision", runtime_precision,
        "--token-ids", args.token_ids,
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-seq-len", str(args.max_seq_len),
        "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        "--telemetry-markers",
    ]
    if args.cublas:
        command.append("--cublas")
    sampler = TelemetrySampler(args.telemetry_interval_ms, not args.no_telemetry)
    with ClockGuard(args.lock_clocks):
        sampler.start()
        try:
            benchmark, markers, diagnostics = run_benchmark_process(command)
        finally:
            sampler.stop()
    validate_benchmark(benchmark)
    measured_tokens = args.repeat * benchmark["tokens"]["generated_per_iteration"]
    report = {
        "runner_schema_version": 1,
        "provenance": provenance,
        "command": command,
        "requested_precision": args.precision,
        "telemetry": summarize_telemetry(sampler, markers, measured_tokens),
        "benchmark": benchmark,
    }
    if diagnostics:
        report["benchmark_diagnostics"] = diagnostics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    console_report = {
        "output": str(args.output),
        "git_commit": provenance["git_commit"],
        "requested_precision": args.precision,
        "statistics": benchmark["statistics"],
        "telemetry": {key: value for key, value in report["telemetry"].items()
                      if key != "samples"},
    }
    print(json.dumps(console_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
