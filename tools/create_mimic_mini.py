from __future__ import annotations

import csv
import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mimic"
DST = ROOT / "mimic-mini"
N_SUBJECTS = 10


def raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def selected_subjects(src: Path, n_subjects: int) -> list[str]:
    image_subjects = {
        p.name[1:]
        for p in (src / "images").iterdir()
        if p.is_dir() and p.name.startswith("p") and p.name[1:].isdigit()
    }
    with (src / "structured" / "patients.csv").open("r", encoding="utf-8", newline="") as f:
        patient_subjects = {row["subject_id"] for row in csv.DictReader(f)}

    selected: list[str] = []
    seen: set[str] = set()
    with (src / "cxr_sampled_with_reports.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row.get("subject_id", "")
            if subject_id in image_subjects and subject_id in patient_subjects and subject_id not in seen:
                selected.append(subject_id)
                seen.add(subject_id)
                if len(selected) == n_subjects:
                    break

    if len(selected) < n_subjects:
        raise RuntimeError(f"Only found {len(selected)} subjects with CXR rows and image folders.")
    return selected


def local_image_path(dst: Path, subject_id: str, study_id: str, dicom_id: str) -> str:
    return str((dst / "images" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg").resolve())


def filter_csv(src_file: Path, dst_file: Path, subject_set: set[str], dst_root: Path, rewrite_cxr_path: bool = False) -> int:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src_file.open("r", encoding="utf-8", errors="replace", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "subject_id" not in reader.fieldnames:
            raise RuntimeError(f"{src_file} does not contain a subject_id column.")

        with dst_file.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("subject_id") not in subject_set:
                    continue
                if rewrite_cxr_path and "imgpath_resolved" in row:
                    row["imgpath_resolved"] = local_image_path(
                        dst_root,
                        row.get("subject_id", ""),
                        row.get("study_id", ""),
                        row.get("dicom_id", ""),
                    )
                writer.writerow(row)
                kept += 1
    return kept


def copy_images(src: Path, dst: Path, subjects: list[str]) -> int:
    copied_files = 0
    dst_images = dst / "images"
    dst_images.mkdir(parents=True, exist_ok=True)
    subject_set = set(subjects)
    for old_dir in dst_images.iterdir():
        if old_dir.is_dir() and old_dir.name.startswith("p") and old_dir.name[1:] not in subject_set:
            shutil.rmtree(old_dir, onexc=make_writable_and_retry)
    for subject_id in subjects:
        src_dir = src / "images" / f"p{subject_id}"
        dst_dir = dst_images / f"p{subject_id}"
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        copied_files += sum(1 for p in dst_dir.rglob("*") if p.is_file())
    return copied_files


def make_writable_and_retry(function, path, _excinfo) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a subject-aligned lightweight MIMIC subset.")
    parser.add_argument("--source", type=Path, default=SRC, help="Source mimic directory.")
    parser.add_argument("--destination", type=Path, default=DST, help="Destination subset directory.")
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS, help="Number of subject IDs to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.source if args.source.is_absolute() else ROOT / args.source
    dst = args.destination if args.destination.is_absolute() else ROOT / args.destination
    n_subjects = args.n_subjects

    raise_csv_limit()
    dst.mkdir(parents=True, exist_ok=True)

    subjects = selected_subjects(src, n_subjects)
    subject_set = set(subjects)
    print(f"Selected {len(subjects)} subjects: {subjects[0]} ... {subjects[-1]}", flush=True)

    summary: dict[str, object] = {
        "source": str(src.resolve()),
        "destination": str(dst.resolve()),
        "n_subjects": len(subjects),
        "subject_ids": subjects,
        "csv_rows": {},
    }

    print("Copying images...", flush=True)
    summary["image_files"] = copy_images(src, dst, subjects)

    jobs = [
        (src / "cxr_sampled_with_reports.csv", dst / "cxr_sampled_with_reports.csv", True),
        (src / "notes" / "discharge_notes.csv", dst / "notes" / "discharge_notes.csv", False),
        (src / "notes" / "radiology_notes.csv", dst / "notes" / "radiology_notes.csv", False),
        (src / "structured" / "admissions.csv", dst / "structured" / "admissions.csv", False),
        (src / "structured" / "diagnoses_icd.csv", dst / "structured" / "diagnoses_icd.csv", False),
        (src / "structured" / "icustays.csv", dst / "structured" / "icustays.csv", False),
        (src / "structured" / "labevents.csv", dst / "structured" / "labevents.csv", False),
        (src / "structured" / "patients.csv", dst / "structured" / "patients.csv", False),
        (src / "structured" / "prescriptions.csv", dst / "structured" / "prescriptions.csv", False),
    ]

    csv_rows: dict[str, int] = {}
    for src_file, dst_file, rewrite_cxr_path in jobs:
        rel = src_file.relative_to(src).as_posix()
        print(f"Filtering {rel}...", flush=True)
        csv_rows[rel] = filter_csv(src_file, dst_file, subject_set, dst, rewrite_cxr_path)
        print(f"  kept {csv_rows[rel]} rows", flush=True)

    summary["csv_rows"] = csv_rows
    with (dst / "mini_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (dst / "subject_ids.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(subjects) + "\n")

    print("Done.", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
