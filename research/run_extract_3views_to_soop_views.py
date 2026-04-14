#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass
class SubjectResult:
    split: str
    subject_id: str
    reason: str
    image_path: str
    mask_path: str
    return_code: int


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[3]
    research_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Batch extract 3-view PNGs into /mnt/disk2/SOOP_views")
    parser.add_argument("--split-dir", type=Path, default=code_root / "datasets" / "fold_raw_trace_fullmodal_mask")
    parser.add_argument("--image-root", type=Path, default=Path("/mnt/disk1/SOOP_TRACE_STRIPPED"))
    parser.add_argument("--mask-root", type=Path, default=Path("/mnt/disk1/SOOP_raw/derivatives/lesion_masks"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/disk1/SOOP_multiview"))
    parser.add_argument("--script-path", type=Path, default=code_root / "utils" / "extract_3views_headless.py")
    parser.add_argument("--log-dir", type=Path, default=research_dir / "logs")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Only validate mapping + file existence, do not extract")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs by rerunning extraction")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Optional limit for quick smoke run")
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_subject_ids(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [str(row["subject_id"]).strip() for row in rows]


def write_failed_csv(path: Path, failed: List[SubjectResult]) -> None:
    fieldnames = ["split", "subject_id", "reason", "image_path", "mask_path", "return_code"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in failed:
            writer.writerow(
                {
                    "split": item.split,
                    "subject_id": item.subject_id,
                    "reason": item.reason,
                    "image_path": item.image_path,
                    "mask_path": item.mask_path,
                    "return_code": item.return_code,
                }
            )


def main() -> None:
    args = parse_args()
    run_id = now_tag()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_log_path = args.log_dir / f"extract_3views_{run_id}.log"
    summary_path = args.log_dir / f"extract_3views_summary_{run_id}.json"
    failed_path = args.log_dir / f"failed_subjects_{run_id}.csv"
    run_manifest_path = args.log_dir / f"run_manifest_{run_id}.json"

    split_files: Dict[str, Path] = {
        "train": args.split_dir / "train.csv",
        "valid": args.split_dir / "valid.csv",
        "test": args.split_dir / "test.csv",
    }

    for split_name, split_file in split_files.items():
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split CSV: {split_name} -> {split_file}")

    if not args.script_path.exists():
        raise FileNotFoundError(f"Missing extractor script: {args.script_path}")

    summary: Dict[str, Dict[str, object]] = {}
    failed: List[SubjectResult] = []

    with run_log_path.open("w", encoding="utf-8") as run_log:
        run_log.write(f"[RUN_ID] {run_id}\n")
        run_log.write(f"[DRY_RUN] {args.dry_run}\n")
        run_log.write(f"[PYTHON_EXE] {args.python_exe}\n")
        run_log.write(f"[SCRIPT] {args.script_path}\n")
        run_log.write(f"[OUTPUT_ROOT] {args.output_root}\n")

        for split_name, split_csv in split_files.items():
            subject_ids = read_subject_ids(split_csv)
            if args.limit_per_split > 0:
                subject_ids = subject_ids[: args.limit_per_split]

            ok = 0
            skipped_no_image = 0
            skipped_no_mask = 0
            skipped_already_exists = 0
            failed_runtime = 0
            success_subjects: List[str] = []

            run_log.write(f"\n[SPLIT] {split_name} total_subjects={len(subject_ids)}\n")

            for subject_id in subject_ids:
                image_path = args.image_root / f"{subject_id}_rec-TRACE_dwi.nii.gz"
                mask_path = args.mask_root / subject_id / "dwi" / f"{subject_id}_space-TRACE_desc-lesion_mask.nii.gz"
                out_dir = args.output_root / split_name / subject_id
                axial_path = out_dir / "Axial.png"
                coronal_path = out_dir / "Coronal.png"
                sagittal_path = out_dir / "Sagittal.png"

                if not image_path.exists():
                    skipped_no_image += 1
                    failed.append(
                        SubjectResult(
                            split=split_name,
                            subject_id=subject_id,
                            reason="missing_image",
                            image_path=str(image_path),
                            mask_path=str(mask_path),
                            return_code=-1,
                        )
                    )
                    run_log.write(f"[SKIP][{split_name}] {subject_id}: missing_image {image_path}\n")
                    continue

                if not mask_path.exists():
                    skipped_no_mask += 1
                    failed.append(
                        SubjectResult(
                            split=split_name,
                            subject_id=subject_id,
                            reason="missing_mask",
                            image_path=str(image_path),
                            mask_path=str(mask_path),
                            return_code=-1,
                        )
                    )
                    run_log.write(f"[SKIP][{split_name}] {subject_id}: missing_mask {mask_path}\n")
                    continue

                if (
                    not args.overwrite
                    and axial_path.exists()
                    and coronal_path.exists()
                    and sagittal_path.exists()
                ):
                    skipped_already_exists += 1
                    ok += 1
                    success_subjects.append(subject_id)
                    run_log.write(f"[SKIP][{split_name}] {subject_id}: already_exists\n")
                    continue

                if args.dry_run:
                    ok += 1
                    success_subjects.append(subject_id)
                    run_log.write(f"[DRY_RUN][{split_name}] {subject_id}: eligible\n")
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    args.python_exe,
                    str(args.script_path),
                    "--image",
                    str(image_path),
                    "--mask",
                    str(mask_path),
                    "--output-dir",
                    str(out_dir),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    ok += 1
                    success_subjects.append(subject_id)
                    run_log.write(f"[OK][{split_name}] {subject_id}\n")
                else:
                    failed_runtime += 1
                    failed.append(
                        SubjectResult(
                            split=split_name,
                            subject_id=subject_id,
                            reason="runtime_error",
                            image_path=str(image_path),
                            mask_path=str(mask_path),
                            return_code=int(proc.returncode),
                        )
                    )
                    run_log.write(f"[FAIL][{split_name}] {subject_id}: return_code={proc.returncode}\n")
                    if proc.stderr:
                        run_log.write(proc.stderr.strip() + "\n")

            summary[split_name] = {
                "total": len(subject_ids),
                "ok": ok,
                "skipped_no_image": skipped_no_image,
                "skipped_no_mask": skipped_no_mask,
                "skipped_already_exists": skipped_already_exists,
                "failed_runtime": failed_runtime,
                "success_subjects": success_subjects,
            }

        summary_payload = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": bool(args.dry_run),
            "paths": {
                "split_dir": str(args.split_dir),
                "image_root": str(args.image_root),
                "mask_root": str(args.mask_root),
                "output_root": str(args.output_root),
                "script_path": str(args.script_path),
                "run_log": str(run_log_path),
                "failed_csv": str(failed_path),
            },
            "summary": summary,
            "totals": {
                "n_failed_records": len(failed),
                "n_ok": int(sum(int(info["ok"]) for info in summary.values())),
                "n_total": int(sum(int(info["total"]) for info in summary.values())),
            },
        }

    write_failed_csv(failed_path, failed)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_executable": args.python_exe,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        },
        "parameters": {
            "dry_run": bool(args.dry_run),
            "overwrite": bool(args.overwrite),
            "limit_per_split": int(args.limit_per_split),
        },
        "artifacts": {
            "run_log": str(run_log_path),
            "summary_json": str(summary_path),
            "failed_subjects_csv": str(failed_path),
        },
    }

    with run_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(run_manifest, file, indent=2, ensure_ascii=False)

    print(f"[OK] Run log: {run_log_path}")
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] Failed list: {failed_path}")
    print(f"[OK] Run manifest: {run_manifest_path}")


if __name__ == "__main__":
    main()
