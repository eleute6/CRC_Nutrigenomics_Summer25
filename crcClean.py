#!/usr/bin/env python3
"""
Merge TCGA HTSeq-Counts files -> log2‑CPM matrix
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

#EDIT THIS ONLY if your folder lives elsewhere
ROOT = Path("/Users/emmaleute/Downloads/GDC_download")

# Accept both compressed and already‑unzipped files
patterns = ["*.htseq.counts.gz", "*.htseq.counts"]

def read_counts(path: Path) -> pd.Series:
    """Return a pandas Series of counts indexed by gene_id."""
    comp = "gzip" if path.suffix == ".gz" else "infer"
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["gene_id", "count"],
        compression=comp,
        dtype={"gene_id": str, "count": "Int64"},   # pandas nullable int
    )
    df = df[~df["gene_id"].str.startswith("__")]         # drop summary rows
    df["gene_id"] = df["gene_id"].str.split(".").str[0]   # strip version suffix
    return df.set_index("gene_id")["count"]

def merge_counts(files: list[Path]) -> pd.DataFrame:
    """Read many HTSeq-Count files and merge into a DataFrame."""
    series = []
    names = []
    for fp in files:
        try:
            s = read_counts(fp)
        except Exception as e:
            logging.error("Failed to parse %s: %s", fp, e)
            continue
        series.append(s)
        names.append(fp.name.split(".")[0])
    if not series:
        raise SystemExit("No valid count files were parsed")
    df = pd.concat(series, axis=1, sort=False)
    df.columns = names
    return df

def main() -> None:
    count_files: list[Path] = []
    for pat in patterns:
        count_files.extend(ROOT.rglob(pat))

    print(f"Found {len(count_files)} files")
    if not count_files:
        raise SystemExit("Nothing matched – check ROOT path & file names")

    # 1. Read & merge
    counts = merge_counts(count_files)
    print("Matrix shape (genes × samples):", counts.shape)

    # 2. Filter low genes
    libsize = counts.sum(axis=0)
    cpm = counts.divide(libsize, axis=1) * 1_000_000
    keep = (cpm >= 1).sum(axis=1) >= 0.20 * cpm.shape[1]
    counts = counts[keep]
    print(f"Kept {keep.sum()} genes after CPM filter")

    # 3. Recompute CPM & log2‑transform
    libsize = counts.sum(axis=0)
    log2_cpm = np.log2(counts.divide(libsize, axis=1) * 1_000_000 + 1)

    # 4. Save
    out = Path.cwd() / "TCGA_COAD_READ_log2CPM.tsv"
    log2_cpm.to_csv(out, sep="\t")
    print("✔️  Saved", out)


if __name__ == "__main__":
    main()
