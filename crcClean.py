#!/usr/bin/env python3
"""Clean TCGA colorectal cancer counts for QVAE models.

This script merges HTSeq count files from TCGA's COAD/READ projects,
filters lowly expressed genes, computes log2-CPM and saves a matrix of
samples x genes suitable for downstream QVAE training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TCGA CRC counts for QVAE simulations"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/tcga_crc"),
        help="Directory containing downloaded HTSeq count files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("TCGA_COAD_READ_log2CPM.tsv"),
        help="Output TSV path",
    )
    return parser.parse_args()


def read_counts(path: Path) -> pd.Series:
    """Return a Series of raw counts indexed by gene ID.

    Supports plain text/TSV as well as simple XML files. The two-column
    layout is assumed to be ``gene_id`` and ``count``; XML files are
    parsed using :mod:`xml.etree.ElementTree` and the first two fields
    are used.
    """

    suffixes = ''.join(path.suffixes).lower()
    comp = 'gzip' if suffixes.endswith('.gz') else 'infer'

    if suffixes.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz', '.htseq.counts', '.htseq.counts.gz')):
        df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['gene_id', 'count'],
            compression=comp,
            dtype={'gene_id': str, 'count': 'Int64'},
        )
    elif suffixes.endswith(('.xml', '.xml.gz')):
        if comp == 'gzip':
            import gzip
            with gzip.open(path, 'rb') as f:
                xml_root = ET.parse(f).getroot()
        else:
            xml_root = ET.parse(path).getroot()
        records = []
        for elem in xml_root:
            values = [child.text for child in list(elem)[:2]]
            records.append(values)
        df = pd.DataFrame(records, columns=['gene_id', 'count'])
    else:
        raise ValueError(f"Unsupported file type: {path}")

    df = df[~df["gene_id"].str.startswith("__")]
    df["gene_id"] = df["gene_id"].str.split(".").str[0]
    return df.set_index("gene_id")["count"]


def merge_counts(paths: list[Path]) -> pd.DataFrame:
    counts = pd.concat([read_counts(p) for p in paths], axis=1, sort=False)
    counts.columns = [p.name.split(".")[0] for p in paths]
    return counts


def filter_low_counts(counts: pd.DataFrame, min_cpm: float = 1.0, frac: float = 0.20) -> pd.DataFrame:
    libsize = counts.sum(axis=0)
    cpm = counts.divide(libsize, axis=1) * 1_000_000
    keep = (cpm >= min_cpm).sum(axis=1) >= frac * counts.shape[1]
    return counts.loc[keep]


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    libsize = counts.sum(axis=0)
    return np.log2(counts.divide(libsize, axis=1) * 1_000_000 + 1)


def main() -> None:
    args = parse_args()

    patterns = [
        "*.htseq.counts",
        "*.htseq.counts.gz",
        "*.tsv",
        "*.tsv.gz",
        "*.txt",
        "*.txt.gz",
        "*.xml",
        "*.xml.gz",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(args.root.rglob(pat))
    if not files:
        raise SystemExit(f"No count files found under {args.root}")

    counts = merge_counts(files)
    counts = filter_low_counts(counts)
    log2cpm = log2_cpm(counts)

    # Machine-learning friendly orientation: samples as rows
    out_matrix = log2cpm.T.sort_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_matrix.to_csv(args.out, sep="\t")
    print(f"Saved {args.out} with shape {out_matrix.shape}")


if __name__ == "__main__":
    main()
