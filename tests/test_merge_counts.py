import logging
import pytest
from pathlib import Path
from crcClean import merge_counts


def write_htseq(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_merge_counts_skips_bad_file(tmp_path, caplog):
    valid = write_htseq(
        tmp_path / "good.htseq.counts",
        "geneA\t1\ngeneB\t2\n__no_feature\t0\n",
    )
    bad = write_htseq(tmp_path / "bad.htseq.counts", "bad\tvalue\n")

    with caplog.at_level(logging.ERROR):
        df = merge_counts([valid, bad])

    assert df.shape[1] == 1
    assert "good" in df.columns
    assert "Failed to parse" in caplog.text


def test_merge_counts_all_bad(tmp_path):
    bad1 = tmp_path / "bad1.htseq.counts"
    bad1.write_text("bad\tvalue\n")

    with pytest.raises(SystemExit):
        merge_counts([bad1])
