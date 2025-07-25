#!/usr/bin/env python3
"""
Merge TCGA HTSeq-Counts files -> log2‑CPM matrix
"""

from pathlib import Path
import pandas as pd
import numpy as np

#EDIT THIS ONLY if your folder lives elsewhere
ROOT = Path("/Users/emmaleute/Downloads/GDC_download")

# Accept both compressed and already‑unzipped files
patterns = ["*.htseq.counts.gz", "*.htseq.counts"]
count_files = []
for pat in patterns:
    count_files.extend(ROOT.rglob(pat))

print(f"Found {len(count_files)} files")
if not count_files:
    raise SystemExit("Nothing matched – check ROOT path & file names")

def read_one_htseq(path: Path) -> pd.Series:
    """Return a pandas Series of counts indexed by gene_id."""
    # autodetect gzip vs plain
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

# 1. Read & merge
counts = pd.concat(
    [read_one_htseq(fp) for fp in count_files],
    axis=1,
    sort=False,
)

counts.columns = [fp.name.split(".")[0] for fp in count_files]
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
