#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

from PIL import Image

EXPECTED_FILES = ("Axial.png", "Coronal.png", "Sagittal.png")


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[3]
    research_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Audit generated SOOP 3-view PNG outputs")
    parser.add_argument("--split-dir", type=Path, default=code_root / "datasets" / "fold_raw_trace")
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/disk2/SOOP_views"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=research_dir / "logs")
    return parser.parse_args()


def find_latest_summary(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("extract_3views_summary_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No summary JSON found in {log_dir}")
    return candidates[-1]


def read_subject_ids(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [str(row["subject_id"]).strip() for row in rows]


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    summary_json = args.summary_json if args.summary_json is not None else find_latest_summary(args.log_dir)
    if not summary_json.exists():
        raise FileNotFoundError(f"Summary JSON does not exist: {summary_json}")

    with summary_json.open("r", encoding="utf-8") as file:
        summary_payload = json.load(file)

    summary = summary_payload.get("summary", {})

    split_files: Dict[str, Path] = {
        "train": args.split_dir / "train.csv",
        "valid": args.split_dir / "valid.csv",
        "test": args.split_dir / "test.csv",
    }

    report: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary_json": str(summary_json),
        "output_root": str(args.output_root),
        "splits": {},
    }

    for split_name, split_csv in split_files.items():
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")

        subjects = read_subject_ids(split_csv)
        split_summary = summary.get(split_name, {}) if isinstance(summary, dict) else {}
        success_subjects = split_summary.get("success_subjects", [])

        if not success_subjects:
            success_subjects = subjects

        missing_files: List[Dict[str, object]] = []
        size_records: List[Dict[str, object]] = []
        widths: List[int] = []
        heights: List[int] = []

        for subject_id in success_subjects:
            subject_dir = args.output_root / split_name / subject_id
            missing_for_subject: List[str] = []

            for file_name in EXPECTED_FILES:
                file_path = subject_dir / file_name
                if not file_path.exists():
                    missing_for_subject.append(file_name)
                    continue

                width, height = image_size(file_path)
                widths.append(width)
                heights.append(height)
                size_records.append(
                    {
                        "subject_id": subject_id,
                        "file": file_name,
                        "width": width,
                        "height": height,
                        "path": str(file_path),
                    }
                )

            if missing_for_subject:
                missing_files.append(
                    {
                        "subject_id": subject_id,
                        "subject_dir": str(subject_dir),
                        "missing": missing_for_subject,
                    }
                )

        width_median = int(median(widths)) if widths else 0
        height_median = int(median(heights)) if heights else 0

        size_warnings: List[Dict[str, object]] = []
        for record in size_records:
            width = int(record["width"])
            height = int(record["height"])
            min_dim = min(width, height)
            max_dim = max(width, height)

            too_small_abs = min_dim < 32
            too_small_rel = width_median > 0 and height_median > 0 and (
                width < int(0.5 * width_median) or height < int(0.5 * height_median)
            )
            too_large_rel = width_median > 0 and height_median > 0 and (
                width > int(2.0 * width_median) or height > int(2.0 * height_median)
            )
            too_aspect_skew = min_dim > 0 and (max_dim / min_dim) > 3.0

            if too_small_abs or too_small_rel or too_large_rel or too_aspect_skew:
                size_warnings.append(
                    {
                        **record,
                        "flags": {
                            "too_small_abs": too_small_abs,
                            "too_small_rel": too_small_rel,
                            "too_large_rel": too_large_rel,
                            "too_aspect_skew": too_aspect_skew,
                        },
                    }
                )

        report["splits"][split_name] = {
            "split_csv": str(split_csv),
            "n_subjects_in_csv": len(subjects),
            "n_subjects_audited": len(success_subjects),
            "n_missing_subject_entries": len(missing_files),
            "n_size_warnings": len(size_warnings),
            "width_median": width_median,
            "height_median": height_median,
            "missing_files": missing_files,
            "size_warnings": size_warnings,
        }

    out_path = args.log_dir / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"[OK] Audit report: {out_path}")


if __name__ == "__main__":
    main()
