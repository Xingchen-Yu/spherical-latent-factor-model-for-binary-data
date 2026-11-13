"""Data loading: the repository's simulated datasets and Voteview .ord roll-call files."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_rdata_matrix(name: str) -> np.ndarray:
    """Load the response matrix stored in ../data/<name>.Rdata (e.g. 'sim_2', 'ymat_eu_3')."""
    import pyreadr

    objs = pyreadr.read_r(str(DATA_DIR / f"{name}.Rdata"))
    key = "ymat" if "ymat" in objs else next(iter(objs))
    return objs[key].to_numpy(dtype=float)


def load_ord(chamber: str, congress: int, max_missing: float = 0.4):
    """Read ../data/{H,S}<congress>_votes.ord as the R code does (readKH2 + ymat_spit).

    Codes 1-3 are Yea, 4-6 Nay, 0 and 7-9 missing; members who miss at least
    ``max_missing`` of the votes are dropped.  Returns (y, names, party) with
    party the 3-digit ICPSR code (100 Democrat, 200 Republican, 328 Independent).
    """
    lines = (DATA_DIR / f"{chamber}{congress}_votes.ord").read_text().splitlines()
    lines = [ln for ln in lines if ln.strip()]
    codes = np.array([[int(c) for c in ln[36:].replace(" ", "")] for ln in lines])
    names = np.array([ln[25:36].strip() for ln in lines])
    party = np.array([int(ln[20:23]) for ln in lines])
    if chamber == "H" and congress == 116:  # as in ymat_spit: first 700 votes, merge Amash's two records
        codes = codes[:, :700]
        codes[201, 429:700] = codes[202, 429:700]
        keep = np.arange(len(lines)) != 202
        codes, names, party = codes[keep], names[keep], party[keep]
    y = np.full(codes.shape, np.nan)
    y[(codes >= 1) & (codes <= 3)] = 1.0
    y[(codes >= 4) & (codes <= 6)] = 0.0
    keep = np.isnan(y).mean(1) < max_missing
    return y[keep], names[keep], party[keep]
