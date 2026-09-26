"""
run_preprocess.py — CLI entry-point for batch preprocessing.

Reads raw source TSVs, cleans & normalizes every row, and writes preprocessed TSVs.

Usage:
    python -m src.preprocessing.run_preprocess \
        --input-dir dataset/train \
        --output-dir dataset/preprocessed/train
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl

try:
    from .normalizer import preprocess_row
except ImportError:
    _this_dir = Path(__file__).resolve().parent
    _pkg_root = _this_dir.parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from src.preprocessing.normalizer import preprocess_row

OUTPUT_COLUMNS = [
    "entity_id",
    "country",
    "business_name",
    "business_address",
    # --- name fields ---
    "name_original",
    "name_clean",
    "name_translit",
    "name_core",
    "legal_suffix",
    # --- address fields ---
    "addr_original",
    "addr_clean",
    "addr_translit",
    "addr_city",
    "addr_state",
    "addr_postal_code",
]


def process_file(input_path: str, output_path: str, chunk_size: int = 100_000) -> int:
    """Read a raw source TSV, preprocess every row, write the result."""
    print(f"  Reading  {input_path} ...")
    t0 = time.perf_counter()

    df = pl.read_csv(
        input_path,
        separator="\t",
        schema_overrides={
            "entity_id": pl.Utf8,
            "business_name": pl.Utf8,
            "business_address": pl.Utf8,
            "country": pl.Utf8,
        },
        null_values=["", "null", "NULL", "nan", "NaN", "None"],
        ignore_errors=True,
    )
    t_read = time.perf_counter() - t0
    n_rows = df.height
    print(f"    -> {n_rows:,} rows read in {t_read:.1f}s")

    results: list[dict[str, str]] = []
    processed = 0

    ids = df["entity_id"].to_list()
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    countries = df["country"].to_list()
    del df

    t1 = time.perf_counter()
    for i in range(n_rows):
        row = preprocess_row(
            entity_id=ids[i] or "",
            business_name=names[i] or "",
            business_address=addrs[i] or "",
            country=countries[i] or "",
        )
        results.append(row)

        processed += 1
        if processed % chunk_size == 0:
            elapsed = time.perf_counter() - t1
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"    ... {processed:>10,} / {n_rows:,}  ({rate:,.0f} rows/s)")

    t_process = time.perf_counter() - t1
    print(f"    -> processed in {t_process:.1f}s ({n_rows / t_process:,.0f} rows/s)")

    t2 = time.perf_counter()
    out_df = pl.DataFrame(results, schema={col: pl.Utf8 for col in OUTPUT_COLUMNS})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.write_csv(output_path, separator="\t")
    t_write = time.perf_counter() - t2
    print(f"    -> written {output_path} in {t_write:.1f}s")

    return n_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess business entity source TSVs."
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing *_source1.tsv, *_source2.tsv, *_source3.tsv",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory where preprocessed *_clean.tsv files will be saved",
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=100_000,
        help="Progress reporting interval (default: 100,000 rows)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(input_dir.glob("*_source*.tsv"))
    if not sources:
        print(f"WARNING: No '*_source*.tsv' files found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sources)} source file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    total_rows = 0
    t_start = time.perf_counter()

    for src_path in sources:
        stem = src_path.stem
        out_name = f"{stem}_clean.tsv"
        out_path = output_dir / out_name

        print(f"\nProcessing {src_path.name} -> {out_name}")
        rows = process_file(
            input_path=str(src_path),
            output_path=str(out_path),
            chunk_size=args.chunk_size,
        )
        total_rows += rows

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"ALL DONE: {total_rows:,} rows across {len(sources)} files in {t_total:.1f}s ({t_total / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
