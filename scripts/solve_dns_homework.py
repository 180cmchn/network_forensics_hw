#!/usr/bin/env python3
"""Solve the three DNS homework submission tasks from extracted CSV data.

The script intentionally keeps all inputs read-only and writes only the three
submission CSVs plus a compact run summary.  Large access/flint tables are
processed in chunks so it can run in the provided virtual environment without
requiring an external database.
"""

from __future__ import annotations

import json
import math
import re
import time
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

nx: Any = importlib.import_module("networkx")
np: Any = importlib.import_module("numpy")
pd: Any = importlib.import_module("pandas")
MiniBatchKMeans: Any = getattr(importlib.import_module("sklearn.cluster"), "MiniBatchKMeans")
TruncatedSVD: Any = getattr(importlib.import_module("sklearn.decomposition"), "TruncatedSVD")
ensemble_module = importlib.import_module("sklearn.ensemble")
ExtraTreesClassifier: Any = getattr(ensemble_module, "ExtraTreesClassifier")
HistGradientBoostingClassifier: Any = getattr(ensemble_module, "HistGradientBoostingClassifier")
TfidfVectorizer: Any = getattr(importlib.import_module("sklearn.feature_extraction.text"), "TfidfVectorizer")
LogisticRegression: Any = getattr(importlib.import_module("sklearn.linear_model"), "LogisticRegression")
metrics_module = importlib.import_module("sklearn.metrics")
f1_score: Any = getattr(metrics_module, "f1_score")
silhouette_score: Any = getattr(metrics_module, "silhouette_score")
StratifiedKFold: Any = getattr(importlib.import_module("sklearn.model_selection"), "StratifiedKFold")
NearestNeighbors: Any = getattr(importlib.import_module("sklearn.neighbors"), "NearestNeighbors")
StandardScaler: Any = getattr(importlib.import_module("sklearn.preprocessing"), "StandardScaler")
compute_sample_weight: Any = getattr(importlib.import_module("sklearn.utils.class_weight"), "compute_sample_weight")
DataFrame = Any
NDArray = Any


RANDOM_STATE = 42
CHUNK_ROWS = 500_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


def log(message: str) -> None:
    """Print a timestamped progress message immediately."""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_output_dirs() -> None:
    for path in [RESULTS_DIR / "dns1", RESULTS_DIR / "dns2", RESULTS_DIR / "dns3", REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text or "unknown"


def entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = defaultdict(int)
    for ch in text:
        counts[ch] += 1
    total = float(len(text))
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def max_run(text: str, char: str) -> int:
    best = 0
    current = 0
    for ch in text:
        if ch == char:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def fqdn_lexical_features(fqdn: DataFrame) -> DataFrame:
    """Build deterministic numeric features from encoded FQDN strings."""

    rows: list[dict[str, float | int | str]] = []
    tlds = fqdn["encoded_fqdn"].fillna("").astype(str).str.rsplit(".", n=1).str[-1]
    tld_counts = tlds.value_counts()
    top_tlds = sorted(tld_counts.index.tolist(), key=lambda item: (-int(tld_counts[item]), str(item)))[:20]
    token_re = re.compile(r"\[[^\]]+\]|[A-Za-z]+|\d+|[-_]+")

    for encoded, fqdn_no in zip(fqdn["encoded_fqdn"].fillna("").astype(str), fqdn["fqdn_no"].astype(str)):
        labels = encoded.split(".") if encoded else [""]
        tld = labels[-1] if labels else ""
        label_lengths = [len(label) for label in labels]
        bracket_tokens = re.findall(r"\[[^\]]+\]", encoded)
        tokens = token_re.findall(encoded)
        char_len = len(encoded)
        digit_count = sum(ch.isdigit() for ch in encoded)
        alpha_count = sum(ch.isalpha() for ch in encoded)
        hyphen_count = encoded.count("-")
        dot_count = encoded.count(".")
        bracket_char_count = encoded.count("[") + encoded.count("]")
        row: dict[str, float | int | str] = {
            "fqdn_no": fqdn_no,
            "lex_len": char_len,
            "lex_label_count": len(labels),
            "lex_dot_count": dot_count,
            "lex_tld_len": len(tld),
            "lex_alpha_count": alpha_count,
            "lex_digit_count": digit_count,
            "lex_zero_count": encoded.count("0"),
            "lex_hyphen_count": hyphen_count,
            "lex_bracket_token_count": len(bracket_tokens),
            "lex_bracket_char_count": bracket_char_count,
            "lex_token_count": len(tokens),
            "lex_unique_char_count": len(set(encoded)),
            "lex_entropy": entropy(encoded),
            "lex_max_label_len": max(label_lengths) if label_lengths else 0,
            "lex_mean_label_len": float(np.mean(label_lengths)) if label_lengths else 0.0,
            "lex_min_label_len": min(label_lengths) if label_lengths else 0,
            "lex_max_a_run": max_run(encoded, "a"),
            "lex_max_zero_run": max_run(encoded, "0"),
            "lex_has_hyphen": int(hyphen_count > 0),
            "lex_has_bracket_word": int(bool(bracket_tokens)),
            "lex_digit_ratio": digit_count / max(char_len, 1),
            "lex_alpha_ratio": alpha_count / max(char_len, 1),
            "lex_hyphen_ratio": hyphen_count / max(char_len, 1),
            "lex_bracket_ratio": bracket_char_count / max(char_len, 1),
            "lex_tld_is_common": int(tld in {"com", "net", "org", "cn", "ru", "ir", "vn"}),
        }
        for top_tld in top_tlds:
            row[f"lex_tld_{safe_name(top_tld)}"] = int(tld == top_tld)
        rows.append(row)

    return pd.DataFrame(rows).set_index("fqdn_no")


def finalize_moments(df: DataFrame, prefix: str, sum_col: str, sumsq_col: str, count_col: str) -> None:
    count = df[count_col].replace(0, np.nan)
    mean = df[sum_col] / count
    variance = (df[sumsq_col] / count) - mean.pow(2)
    df[f"{prefix}_mean"] = mean.fillna(0.0)
    df[f"{prefix}_std"] = np.sqrt(np.maximum(variance.fillna(0.0), 0.0))


def aggregate_date_pairs(day_frames: list[DataFrame], prefix: str) -> DataFrame:
    if not day_frames:
        return pd.DataFrame()

    days = pd.concat(day_frames, ignore_index=True).dropna()
    if days.empty:
        return pd.DataFrame()
    days["_date"] = days["_date"].astype(str).str.slice(0, 8)
    days = days.drop_duplicates(["fqdn_no", "_date"])
    grouped = days.groupby("fqdn_no")["_date"]
    out = pd.DataFrame(
        {
            f"{prefix}_active_days": grouped.nunique(),
            f"{prefix}_first_date_num": pd.to_numeric(grouped.min(), errors="coerce"),
            f"{prefix}_last_date_num": pd.to_numeric(grouped.max(), errors="coerce"),
        }
    )

    first_dt = pd.to_datetime(out[f"{prefix}_first_date_num"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    last_dt = pd.to_datetime(out[f"{prefix}_last_date_num"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    out[f"{prefix}_span_days"] = (last_dt - first_dt).dt.days.fillna(0).clip(lower=0)
    out[f"{prefix}_active_rate"] = out[f"{prefix}_active_days"] / (out[f"{prefix}_span_days"] + 1.0)
    return out


def aggregate_access(
    path: Path,
    request_col: str,
    fqdn_col: str = "fqdn_no",
    ip_col: str | None = None,
    ip_count_col: str | None = None,
    total_request_col: str | None = None,
    date_col: str | None = None,
    hour_col: str | None = None,
    time_col: str | None = None,
    prefix: str = "access",
) -> DataFrame:
    """Aggregate access rows by domain using chunked reads."""

    log(f"Aggregating {path.relative_to(PROJECT_ROOT)} in chunks")
    usecols = [fqdn_col, request_col]
    for optional in [ip_col, ip_count_col, total_request_col, date_col, hour_col, time_col]:
        if optional and optional not in usecols:
            usecols.append(optional)

    moment_frames: list[DataFrame] = []
    hour_frames: list[DataFrame] = []
    day_frames: list[DataFrame] = []
    ip_frames: list[DataFrame] = []
    ratio_frames: list[DataFrame] = []

    for chunk_no, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS), start=1):
        chunk = chunk.rename(columns={fqdn_col: "fqdn_no"})
        chunk["fqdn_no"] = chunk["fqdn_no"].astype(str)
        req = pd.to_numeric(chunk[request_col], errors="coerce").fillna(0.0).astype(float)
        chunk["_req"] = req
        chunk["_req2"] = req * req
        grouped = chunk.groupby("fqdn_no", sort=False)
        part = pd.DataFrame(
            {
                f"{prefix}_rows": grouped.size(),
                f"{prefix}_req_sum": grouped["_req"].sum(),
                f"{prefix}_req_sumsq": grouped["_req2"].sum(),
                f"{prefix}_req_max": grouped["_req"].max(),
                f"{prefix}_req_min": grouped["_req"].min(),
            }
        )
        if ip_count_col:
            ip_count = pd.to_numeric(chunk[ip_count_col], errors="coerce").fillna(0.0).astype(float)
            chunk["_ip_count"] = ip_count
            chunk["_ip_count2"] = ip_count * ip_count
            grouped = chunk.groupby("fqdn_no", sort=False)
            part[f"{prefix}_ip_count_sum"] = grouped["_ip_count"].sum()
            part[f"{prefix}_ip_count_sumsq"] = grouped["_ip_count2"].sum()
            part[f"{prefix}_ip_count_max"] = grouped["_ip_count"].max()
        moment_frames.append(part)

        if total_request_col:
            total = pd.to_numeric(chunk[total_request_col], errors="coerce").replace(0, np.nan).astype(float)
            ratio = (req / total).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            chunk["_ratio"] = ratio
            chunk["_ratio2"] = ratio * ratio
            grouped = chunk.groupby("fqdn_no", sort=False)
            ratio_frames.append(
                pd.DataFrame(
                    {
                        f"{prefix}_global_ratio_sum": grouped["_ratio"].sum(),
                        f"{prefix}_global_ratio_sumsq": grouped["_ratio2"].sum(),
                        f"{prefix}_global_ratio_max": grouped["_ratio"].max(),
                    }
                )
            )

        if time_col:
            time_text = chunk[time_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(14)
            chunk["_date"] = time_text.str.slice(0, 8)
            chunk["_hour"] = pd.to_numeric(time_text.str.slice(8, 10), errors="coerce")
        elif date_col:
            chunk["_date"] = chunk[date_col].astype(str).str.replace(r"\.0$", "", regex=True).str.slice(0, 8)
            if hour_col:
                chunk["_hour"] = pd.to_numeric(chunk[hour_col], errors="coerce")

        if "_date" in chunk.columns:
            day_frames.append(chunk[["fqdn_no", "_date"]].drop_duplicates())
        if "_hour" in chunk.columns:
            hour_req = chunk.dropna(subset=["_hour"]).copy()
            hour_req["_hour"] = hour_req["_hour"].astype(int).clip(0, 23)
            h = hour_req.groupby(["fqdn_no", "_hour"], sort=False)["_req"].sum().unstack(fill_value=0.0)
            h = h.rename(columns={col: f"{prefix}_hour_req_{int(col):02d}" for col in h.columns})
            hour_frames.append(h)

        if ip_col:
            ip_count = chunk.groupby("fqdn_no", sort=False)[ip_col].nunique(dropna=True)
            ip_frames.append(ip_count.to_frame(f"{prefix}_ip_unique_chunked"))

        if chunk_no % 5 == 0:
            log(f"  processed {chunk_no * CHUNK_ROWS:,}+ access rows from {path.name}")

    if not moment_frames:
        return pd.DataFrame()

    agg_map = {
        f"{prefix}_rows": "sum",
        f"{prefix}_req_sum": "sum",
        f"{prefix}_req_sumsq": "sum",
        f"{prefix}_req_max": "max",
        f"{prefix}_req_min": "min",
    }
    if ip_count_col:
        agg_map.update(
            {
                f"{prefix}_ip_count_sum": "sum",
                f"{prefix}_ip_count_sumsq": "sum",
                f"{prefix}_ip_count_max": "max",
            }
        )
    out = pd.concat(moment_frames).groupby(level=0).agg(agg_map)
    finalize_moments(out, f"{prefix}_req", f"{prefix}_req_sum", f"{prefix}_req_sumsq", f"{prefix}_rows")
    out[f"{prefix}_burst_ratio"] = out[f"{prefix}_req_max"] / out[f"{prefix}_req_sum"].replace(0, np.nan)
    out[f"{prefix}_req_per_row"] = out[f"{prefix}_req_sum"] / out[f"{prefix}_rows"].replace(0, np.nan)
    if ip_count_col:
        finalize_moments(out, f"{prefix}_ip_count", f"{prefix}_ip_count_sum", f"{prefix}_ip_count_sumsq", f"{prefix}_rows")

    if ratio_frames:
        ratio = pd.concat(ratio_frames).groupby(level=0).agg(
            {
                f"{prefix}_global_ratio_sum": "sum",
                f"{prefix}_global_ratio_sumsq": "sum",
                f"{prefix}_global_ratio_max": "max",
            }
        )
        out = out.join(ratio, how="left")
        finalize_moments(out, f"{prefix}_global_ratio", f"{prefix}_global_ratio_sum", f"{prefix}_global_ratio_sumsq", f"{prefix}_rows")

    if hour_frames:
        hours = pd.concat(hour_frames).groupby(level=0).sum()
        for hour in range(24):
            col = f"{prefix}_hour_req_{hour:02d}"
            if col not in hours.columns:
                hours[col] = 0.0
        hours = hours[[f"{prefix}_hour_req_{hour:02d}" for hour in range(24)]]
        hour_sum = hours.sum(axis=1).replace(0, np.nan)
        for hour in range(24):
            hours[f"{prefix}_hour_frac_{hour:02d}"] = hours[f"{prefix}_hour_req_{hour:02d}"] / hour_sum
        hours[f"{prefix}_active_hours"] = (hours[[f"{prefix}_hour_req_{hour:02d}" for hour in range(24)]] > 0).sum(axis=1)
        hours[f"{prefix}_night_frac"] = hours[[f"{prefix}_hour_req_{hour:02d}" for hour in list(range(0, 6)) + [23]]].sum(axis=1) / hour_sum
        hours[f"{prefix}_workhour_frac"] = hours[[f"{prefix}_hour_req_{hour:02d}" for hour in range(8, 19)]].sum(axis=1) / hour_sum
        out = out.join(hours, how="left")

    dates = aggregate_date_pairs(day_frames, prefix)
    if not dates.empty:
        out = out.join(dates, how="left")
        out[f"{prefix}_req_per_active_day"] = out[f"{prefix}_req_sum"] / out[f"{prefix}_active_days"].replace(0, np.nan)

    if ip_frames:
        ips = pd.concat(ip_frames).groupby(level=0).sum()
        out = out.join(ips, how="left")
        out[f"{prefix}_req_per_unique_ip_chunked"] = out[f"{prefix}_req_sum"] / out[f"{prefix}_ip_unique_chunked"].replace(0, np.nan)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def connected_component_features(edges: DataFrame, source_col: str, target_col: str, prefix: str) -> DataFrame:
    if edges.empty:
        return pd.DataFrame()
    graph = nx.Graph()
    graph.add_edges_from(edges[[source_col, target_col]].itertuples(index=False, name=None))
    comp_size: dict[str, int] = {}
    for component in nx.connected_components(graph):
        size = len(component)
        for node in component:
            comp_size[str(node)] = size
    degree = {str(node): int(value) for node, value in graph.degree()}
    nodes = sorted(set(edges[source_col].astype(str)))
    return pd.DataFrame(
        {
            f"{prefix}_cname_graph_degree": [degree.get(node, 0) for node in nodes],
            f"{prefix}_cname_component_size": [comp_size.get(node, 1) for node in nodes],
        },
        index=nodes,
    )


def aggregate_flint(
    path: Path,
    fqdn_col: str = "fqdn_no",
    ttl_col: str | None = None,
    prefix: str = "flint",
    return_pairs: bool = False,
) -> tuple[DataFrame, DataFrame | None]:
    """Aggregate DNS answer rows by source domain using chunked reads."""

    log(f"Aggregating {path.relative_to(PROJECT_ROOT)} in chunks")
    usecols = [fqdn_col, "flintType", "encoded_value", "requestCnt", "date"]
    if ttl_col:
        usecols.append(ttl_col)

    moment_frames: list[DataFrame] = []
    type_req_frames: list[DataFrame] = []
    type_row_frames: list[DataFrame] = []
    day_frames: list[DataFrame] = []
    pair_frames: list[DataFrame] = []
    ttl_frames: list[DataFrame] = []
    cname_frames: list[DataFrame] = []

    for chunk_no, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS), start=1):
        chunk = chunk.rename(columns={fqdn_col: "fqdn_no"})
        chunk["fqdn_no"] = chunk["fqdn_no"].astype(str)
        chunk["encoded_value"] = chunk["encoded_value"].astype(str)
        req = pd.to_numeric(chunk["requestCnt"], errors="coerce").fillna(0.0).astype(float)
        chunk["_req"] = req
        chunk["_req2"] = req * req
        grouped = chunk.groupby("fqdn_no", sort=False)
        moment_frames.append(
            pd.DataFrame(
                {
                    f"{prefix}_rows": grouped.size(),
                    f"{prefix}_req_sum": grouped["_req"].sum(),
                    f"{prefix}_req_sumsq": grouped["_req2"].sum(),
                    f"{prefix}_req_max": grouped["_req"].max(),
                    f"{prefix}_req_min": grouped["_req"].min(),
                }
            )
        )

        type_req = chunk.groupby(["fqdn_no", "flintType"], sort=False)["_req"].sum().unstack(fill_value=0.0)
        type_req = type_req.rename(columns={col: f"{prefix}_type_{safe_name(col)}_req" for col in type_req.columns})
        type_req_frames.append(type_req)
        type_rows = chunk.groupby(["fqdn_no", "flintType"], sort=False).size().unstack(fill_value=0)
        type_rows = type_rows.astype(float)
        type_rows = type_rows.rename(columns={col: f"{prefix}_type_{safe_name(col)}_rows" for col in type_rows.columns})
        type_row_frames.append(type_rows)

        day = chunk[["fqdn_no", "date"]].drop_duplicates().rename(columns={"date": "_date"})
        day_frames.append(day)

        pairs = chunk[["fqdn_no", "encoded_value"]].drop_duplicates()
        pair_frames.append(pairs)
        cname = pairs[pairs["encoded_value"].str.startswith("fqdn_", na=False)]
        if not cname.empty:
            cname_frames.append(cname)

        if ttl_col:
            ttl = pd.to_numeric(chunk[ttl_col], errors="coerce")
            chunk["_ttl"] = ttl.fillna(0.0).astype(float)
            chunk["_ttl2"] = chunk["_ttl"] * chunk["_ttl"]
            grouped = chunk.groupby("fqdn_no", sort=False)
            ttl_frames.append(
                pd.DataFrame(
                    {
                        f"{prefix}_ttl_sum": grouped["_ttl"].sum(),
                        f"{prefix}_ttl_sumsq": grouped["_ttl2"].sum(),
                        f"{prefix}_ttl_min": grouped["_ttl"].min(),
                        f"{prefix}_ttl_max": grouped["_ttl"].max(),
                        f"{prefix}_ttl_unique_chunked": grouped["_ttl"].nunique(),
                    }
                )
            )

        if chunk_no % 5 == 0:
            log(f"  processed {chunk_no * CHUNK_ROWS:,}+ flint rows from {path.name}")

    if not moment_frames:
        return pd.DataFrame(), None

    out = pd.concat(moment_frames).groupby(level=0).agg(
        {
            f"{prefix}_rows": "sum",
            f"{prefix}_req_sum": "sum",
            f"{prefix}_req_sumsq": "sum",
            f"{prefix}_req_max": "max",
            f"{prefix}_req_min": "min",
        }
    )
    finalize_moments(out, f"{prefix}_req", f"{prefix}_req_sum", f"{prefix}_req_sumsq", f"{prefix}_rows")
    out[f"{prefix}_burst_ratio"] = out[f"{prefix}_req_max"] / out[f"{prefix}_req_sum"].replace(0, np.nan)

    for frames in [type_req_frames, type_row_frames]:
        if frames:
            typed = pd.concat(frames).groupby(level=0).sum()
            out = out.join(typed, how="left")

    dates = aggregate_date_pairs(day_frames, prefix)
    if not dates.empty:
        out = out.join(dates, how="left")
        out[f"{prefix}_req_per_active_day"] = out[f"{prefix}_req_sum"] / out[f"{prefix}_active_days"].replace(0, np.nan)

    pairs_all = pd.concat(pair_frames, ignore_index=True).drop_duplicates(["fqdn_no", "encoded_value"])
    value_counts = pairs_all.groupby("encoded_value")["fqdn_no"].nunique().rename("target_domain_degree")
    pair_degree = pairs_all.join(value_counts, on="encoded_value")
    share = pair_degree.groupby("fqdn_no")["target_domain_degree"].agg(["size", "mean", "max", "sum"])
    share = share.rename(
        columns={
            "size": f"{prefix}_target_unique",
            "mean": f"{prefix}_target_degree_mean",
            "max": f"{prefix}_target_degree_max",
            "sum": f"{prefix}_target_degree_sum",
        }
    )
    pair_degree["_target_is_fqdn"] = pair_degree["encoded_value"].str.startswith("fqdn_", na=False).astype(int)
    pair_degree["_target_is_ip_like"] = 1 - pair_degree["_target_is_fqdn"]
    target_kind = pair_degree.groupby("fqdn_no")[["_target_is_fqdn", "_target_is_ip_like"]].sum()
    target_kind = target_kind.rename(
        columns={
            "_target_is_fqdn": f"{prefix}_target_fqdn_count",
            "_target_is_ip_like": f"{prefix}_target_ip_like_count",
        }
    )
    out = out.join(share, how="left").join(target_kind, how="left")
    out[f"{prefix}_target_shared_frac"] = (
        (out[f"{prefix}_target_degree_sum"] - out[f"{prefix}_target_unique"]) / out[f"{prefix}_target_degree_sum"].replace(0, np.nan)
    )

    if cname_frames:
        cname_edges = pd.concat(cname_frames, ignore_index=True).drop_duplicates(["fqdn_no", "encoded_value"])
        graph_feats = connected_component_features(cname_edges, "fqdn_no", "encoded_value", prefix)
        out = out.join(graph_feats, how="left")

    if ttl_frames:
        ttl_out = pd.concat(ttl_frames).groupby(level=0).agg(
            {
                f"{prefix}_ttl_sum": "sum",
                f"{prefix}_ttl_sumsq": "sum",
                f"{prefix}_ttl_min": "min",
                f"{prefix}_ttl_max": "max",
                f"{prefix}_ttl_unique_chunked": "sum",
            }
        )
        out = out.join(ttl_out, how="left")
        finalize_moments(out, f"{prefix}_ttl", f"{prefix}_ttl_sum", f"{prefix}_ttl_sumsq", f"{prefix}_rows")
        out[f"{prefix}_ttl_low_frac_proxy"] = (out[f"{prefix}_ttl_min"] <= 300).astype(int)
        out[f"{prefix}_ttl_high_frac_proxy"] = (out[f"{prefix}_ttl_max"] >= 86400).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, pairs_all if return_pairs else None


def aggregate_ip_metadata(ip_paths: Iterable[Path], pairs: DataFrame, prefix: str = "ipmeta") -> DataFrame:
    """Join flint target values to lightweight IP metadata features."""

    if pairs.empty:
        return pd.DataFrame()
    target_values = set(pairs.loc[~pairs["encoded_value"].str.startswith("fqdn_", na=False), "encoded_value"].astype(str).unique())
    if not target_values:
        return pd.DataFrame()

    log(f"Building IP metadata features for {len(target_values):,} DNS target values")
    ip_frames: list[DataFrame] = []
    usecols = ["encoded_ip", "country", "subdivision", "city", "latitude", "longitude", "isp"]
    for path in ip_paths:
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS):
            chunk["encoded_ip"] = chunk["encoded_ip"].astype(str)
            keep = chunk[chunk["encoded_ip"].isin(target_values)].copy()
            if not keep.empty:
                ip_frames.append(keep)
    if not ip_frames:
        return pd.DataFrame()

    ip_info = pd.concat(ip_frames, ignore_index=True).drop_duplicates("encoded_ip")
    joined = pairs.merge(ip_info, left_on="encoded_value", right_on="encoded_ip", how="inner")
    if joined.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=sorted(joined["fqdn_no"].astype(str).unique()))
    grouped = joined.groupby("fqdn_no")
    for col in ["country", "subdivision", "city", "isp"]:
        out[f"{prefix}_{col}_nunique"] = grouped[col].nunique(dropna=True)
    for col in ["latitude", "longitude"]:
        numeric = pd.to_numeric(joined[col], errors="coerce")
        tmp = joined[["fqdn_no"]].copy()
        tmp[col] = numeric
        out[f"{prefix}_{col}_mean"] = tmp.groupby("fqdn_no")[col].mean()
        out[f"{prefix}_{col}_std"] = tmp.groupby("fqdn_no")[col].std()
    known_counts = grouped.size().rename(f"{prefix}_known_target_count")
    out = out.join(known_counts, how="left")
    country_counts = joined.groupby(["fqdn_no", "country"]).size()
    dominant_country = country_counts.groupby(level=0).max().rename(f"{prefix}_dominant_country_count")
    out = out.join(dominant_country, how="left")
    out[f"{prefix}_dominant_country_frac"] = out[f"{prefix}_dominant_country_count"] / out[f"{prefix}_known_target_count"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def whois_features(path: Path) -> DataFrame:
    """Parse dns1 WHOIS JSON into compact per-domain features."""

    log(f"Parsing WHOIS features from {path.relative_to(PROJECT_ROOT)}")
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    ref_ms = pd.Timestamp("2020-05-31").value // 1_000_000
    acc: dict[str, dict[str, Any]] = {}
    date_fields = ["createddate", "expiresdate", "updateddate"]

    def entry_for(fqdn_no: str) -> dict[str, Any]:
        if fqdn_no not in acc:
            acc[fqdn_no] = {
                "whois_record_count": 0,
                "whois_ns_count_sum": 0,
                "whois_ns_count_max": 0,
                "whois_has_admin_email": 0,
                "whois_has_registrant_email": 0,
                "whois_has_tech_email": 0,
                "whois_has_country": 0,
                "whois_nameservers": set(),
                "whois_servers": set(),
            }
            for field in date_fields:
                acc[fqdn_no][f"{field}_min"] = math.inf
                acc[fqdn_no][f"{field}_max"] = -math.inf
                acc[fqdn_no][f"{field}_count"] = 0
        return acc[fqdn_no]

    for record in records:
        fqdn_no = str(record.get("fqdn_no", ""))
        if not fqdn_no:
            continue
        item = entry_for(fqdn_no)
        item["whois_record_count"] = int(item["whois_record_count"]) + 1
        nameservers = record.get("nameservers") or []
        if isinstance(nameservers, str):
            nameservers = [nameservers]
        ns_set = item["whois_nameservers"]
        assert isinstance(ns_set, set)
        ns_set.update(str(ns).lower() for ns in nameservers if ns)
        item["whois_ns_count_sum"] = int(item["whois_ns_count_sum"]) + len(nameservers)
        item["whois_ns_count_max"] = max(int(item["whois_ns_count_max"]), len(nameservers))
        server = record.get("whoisserver")
        server_set = item["whois_servers"]
        assert isinstance(server_set, set)
        if server:
            server_set.add(str(server).lower())

        for email_field in ["admin_email", "registrant_email", "tech_email"]:
            if record.get(email_field):
                item[f"whois_has_{email_field}"] = 1
        if record.get("registrant_country"):
            item["whois_has_country"] = 1
        for field in date_fields:
            value = record.get(field)
            if value is None or pd.isna(value):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric <= 0:
                continue
            item[f"{field}_min"] = min(float(item[f"{field}_min"]), numeric)
            item[f"{field}_max"] = max(float(item[f"{field}_max"]), numeric)
            item[f"{field}_count"] = int(item[f"{field}_count"]) + 1

    rows: list[dict[str, float | int | str]] = []
    for fqdn_no, item in acc.items():
        row: dict[str, float | int | str] = {"fqdn_no": fqdn_no}
        record_count = max(int(item["whois_record_count"]), 1)
        row["whois_record_count"] = record_count
        row["whois_ns_count_mean"] = int(item["whois_ns_count_sum"]) / record_count
        row["whois_ns_count_max"] = int(item["whois_ns_count_max"])
        ns_set = item["whois_nameservers"]
        server_set = item["whois_servers"]
        assert isinstance(ns_set, set)
        assert isinstance(server_set, set)
        row["whois_nameserver_nunique"] = len(ns_set)
        row["whois_server_nunique"] = len(server_set)
        for field in ["whois_has_admin_email", "whois_has_registrant_email", "whois_has_tech_email", "whois_has_country"]:
            row[field] = int(item[field])
        created_min = float(item["createddate_min"])
        expires_max = float(item["expiresdate_max"])
        updated_max = float(item["updateddate_max"])
        row["whois_created_age_days"] = (ref_ms - created_min) / 86_400_000 if math.isfinite(created_min) else 0.0
        row["whois_expires_after_days"] = (expires_max - ref_ms) / 86_400_000 if math.isfinite(expires_max) else 0.0
        row["whois_updated_age_days"] = (ref_ms - updated_max) / 86_400_000 if math.isfinite(updated_max) else 0.0
        row["whois_registration_span_days"] = (expires_max - created_min) / 86_400_000 if math.isfinite(created_min) and math.isfinite(expires_max) else 0.0
        for field in date_fields:
            row[f"whois_{field}_count"] = int(item[f"{field}_count"])
        rows.append(row)

    return pd.DataFrame(rows).set_index("fqdn_no").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def join_feature_blocks(fqdn: DataFrame, blocks: Iterable[DataFrame]) -> DataFrame:
    features = pd.DataFrame(index=fqdn["fqdn_no"].astype(str))
    for block in blocks:
        if block is not None and not block.empty:
            features = features.join(block, how="left")
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_dense_matrix(
    fqdn: DataFrame,
    numeric_features: DataFrame,
    text_components: int = 96,
    max_text_features: int = 8000,
) -> tuple[NDArray, list[str]]:
    """Combine scaled numeric features and encoded-FQDN char n-gram SVD."""

    fqdn_order = fqdn["fqdn_no"].astype(str).tolist()
    numeric = numeric_features.reindex(fqdn_order).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not numeric.empty:
        variances = numeric.var(axis=0)
        numeric = numeric.loc[:, variances > 1e-12]
    if numeric.empty:
        numeric_scaled = np.empty((len(fqdn_order), 0), dtype=np.float32)
        names: list[str] = []
    else:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric.to_numpy(dtype=np.float64)).astype(np.float32)
        names = numeric.columns.tolist()

    texts = fqdn["encoded_fqdn"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_text_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(text_components, max(2, tfidf.shape[1] - 1), max(2, len(fqdn_order) - 1))
    if tfidf.shape[1] <= 2 or n_components <= 1:
        text_scaled = np.empty((len(fqdn_order), 0), dtype=np.float32)
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        text_dense = svd.fit_transform(tfidf)
        text_scaled = StandardScaler().fit_transform(text_dense).astype(np.float32)
        names.extend([f"char_svd_{idx:03d}" for idx in range(text_scaled.shape[1])])

    return np.hstack([numeric_scaled, text_scaled]).astype(np.float32), names


def probability_for_class(model: Any, x: NDArray, positive_class: int = 1) -> NDArray:
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_"))
    if positive_class not in classes:
        return np.zeros(x.shape[0], dtype=float)
    return proba[:, classes.index(positive_class)]


def aligned_proba(model: Any, x: NDArray, classes: NDArray) -> NDArray:
    proba = model.predict_proba(x)
    model_classes = list(getattr(model, "classes_"))
    out = np.zeros((x.shape[0], len(classes)), dtype=float)
    for idx, cls in enumerate(classes):
        if cls in model_classes:
            out[:, idx] = proba[:, model_classes.index(cls)]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / len(classes)), where=row_sum > 0)


def rank_percentile(scores: NDArray) -> NDArray:
    series = pd.Series(scores)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def make_binary_model_factories() -> list[Callable[[], Any]]:
    return [
        lambda: ExtraTreesClassifier(
            n_estimators=420,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        lambda: LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        lambda: HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=220,
            l2_regularization=0.08,
            min_samples_leaf=12,
            random_state=RANDOM_STATE,
        ),
    ]


def fit_binary_ensemble(x: NDArray, y: NDArray) -> list[Any]:
    models: list[Any] = []
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    for factory in make_binary_model_factories():
        model = factory()
        if isinstance(model, HistGradientBoostingClassifier):
            model.fit(x, y, sample_weight=sample_weight)
        else:
            model.fit(x, y)
        models.append(model)
    return models


def predict_binary_ensemble(models: list[Any], x: NDArray) -> NDArray:
    if not models:
        return np.zeros(x.shape[0], dtype=float)
    probs = [probability_for_class(model, x, 1) for model in models]
    return np.mean(probs, axis=0)


def best_f1_threshold(y_true: NDArray, proba: NDArray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def solve_dns2(summary: list[str]) -> None:
    log("Starting dns2 supervised binary classification")
    base = DATA_DIR / "dns2" / "test" / "5_question_test"
    train_base = DATA_DIR / "dns2" / "train"
    fqdn = pd.read_csv(base / "fqdn.csv")
    labels = pd.read_csv(train_base / "label.csv")
    fqdn["fqdn_no"] = fqdn["fqdn_no"].astype(str)
    labels["fqdn_no"] = labels["fqdn_no"].astype(str)

    access = aggregate_access(
        base / "access.csv",
        request_col="request_cnt",
        ip_count_col="ip_cnt",
        total_request_col="total_request",
        date_col="date",
        hour_col="hour",
        prefix="dns2_access",
    )
    flint, _ = aggregate_flint(base / "flint.csv", prefix="dns2_flint")
    features = join_feature_blocks(fqdn, [fqdn_lexical_features(fqdn), access, flint])
    x_all, feature_names = build_dense_matrix(fqdn, features, text_components=96, max_text_features=7000)
    index = pd.Series(np.arange(len(fqdn)), index=fqdn["fqdn_no"].astype(str))
    train_positions = labels["fqdn_no"].map(index).dropna().astype(int).to_numpy()
    y = labels.set_index("fqdn_no").loc[fqdn.iloc[train_positions]["fqdn_no"], "label"].astype(int).to_numpy()
    x_train = x_all[train_positions]

    min_class = int(pd.Series(y).value_counts().min())
    splits = min(5, max(2, min_class))
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, va) in enumerate(skf.split(x_train, y), start=1):
        models = fit_binary_ensemble(x_train[tr], y[tr])
        oof[va] = predict_binary_ensemble(models, x_train[va])
        log(f"  dns2 fold {fold}/{splits} complete")
    threshold, cv_f1 = best_f1_threshold(y, oof)

    models = fit_binary_ensemble(x_train, y)
    proba = predict_binary_ensemble(models, x_all)
    pred = (proba >= threshold).astype(int)
    known = labels.set_index("fqdn_no")["label"].astype(int)
    known_mask = fqdn["fqdn_no"].isin(known.index).to_numpy()
    pred[known_mask] = fqdn.loc[known_mask, "fqdn_no"].map(known).astype(int).to_numpy()

    out = pd.DataFrame({"fqdn_no": fqdn["fqdn_no"], "label": pred.astype(int)})
    out.to_csv(RESULTS_DIR / "dns2" / "label.csv", index=False)
    positive_rate = float(out["label"].mean())
    summary.append(
        f"## dns2\n- Rows predicted: {len(out):,}\n- Training labels: {len(labels):,} "
        f"(positive={int(labels['label'].sum()):,})\n- Dense features: {x_all.shape[1]:,} "
        f"({len(feature_names):,} named numeric/SVD columns)\n- CV F1: {cv_f1:.4f} at threshold {threshold:.3f}\n"
        f"- Submitted positive rate: {positive_rate:.3f}\n"
    )
    log(f"dns2 wrote {len(out):,} predictions; CV F1={cv_f1:.4f}, threshold={threshold:.3f}")


def family_oof_score(x: NDArray, y: NDArray) -> float:
    counts = pd.Series(y).value_counts()
    splits = min(4, int(counts.min()))
    if splits < 2:
        return 0.0
    classes = np.sort(np.unique(y))
    oof = np.zeros((len(y), len(classes)), dtype=float)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(x, y):
        tree = ExtraTreesClassifier(
            n_estimators=420,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        logreg = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE)
        tree.fit(x[tr], y[tr])
        logreg.fit(x[tr], y[tr])
        oof[va] = 0.55 * aligned_proba(tree, x[va], classes) + 0.45 * aligned_proba(logreg, x[va], classes)
    pred = classes[np.argmax(oof, axis=1)]
    return float(f1_score(y, pred, average="macro", zero_division=0))


def solve_dns1(summary: list[str]) -> None:
    log("Starting dns1 positive-unlabeled malicious family discovery")
    base = DATA_DIR / "dns1" / "dns1" / "question" / "4_question"
    fqdn = pd.read_csv(base / "fqdn.csv")
    labels = pd.read_csv(base / "label.csv")
    fqdn["fqdn_no"] = fqdn["fqdn_no"].astype(str)
    labels["fqdn_no"] = labels["fqdn_no"].astype(str)

    access = aggregate_access(
        base / "access.csv",
        request_col="count",
        ip_col="encoded_ip",
        time_col="time",
        prefix="dns1_access",
    )
    flint, pairs = aggregate_flint(base / "flint.csv", fqdn_col="fqdn_no_x", prefix="dns1_flint", return_pairs=True)
    ipmeta = aggregate_ip_metadata([base / "ip.csv", base / "ipv6.csv"], pairs if pairs is not None else pd.DataFrame(), prefix="dns1_ipmeta")
    whois = whois_features(base / "whois.json")
    features = join_feature_blocks(fqdn, [fqdn_lexical_features(fqdn), access, flint, ipmeta, whois])
    x_all, feature_names = build_dense_matrix(fqdn, features, text_components=112, max_text_features=8500)

    label_map = labels.set_index("fqdn_no")["family_no"].astype(int)
    fqdn_ids = fqdn["fqdn_no"].astype(str).to_numpy()
    known_mask = np.isin(fqdn_ids, label_map.index.to_numpy())
    y_noisy = known_mask.astype(int)

    noisy_models = fit_binary_ensemble(x_all, y_noisy)
    noisy_score = predict_binary_ensemble(noisy_models, x_all)

    pos_x = x_all[known_mask]
    neighbors = min(10, len(pos_x))
    nn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    nn.fit(pos_x)
    distances = nn.kneighbors(x_all, return_distance=True)[0].mean(axis=1)
    nn_score = rank_percentile(-distances)

    unlabeled_mask = ~known_mask
    unlabeled_scores = noisy_score[unlabeled_mask]
    unlabeled_nn = nn_score[unlabeled_mask]
    reliable_negative_mask = np.zeros(len(fqdn), dtype=bool)
    if len(unlabeled_scores):
        score_cut = float(np.quantile(unlabeled_scores, 0.55))
        nn_cut = float(np.quantile(unlabeled_nn, 0.55))
        reliable_negative_mask = unlabeled_mask & (noisy_score <= score_cut) & (nn_score <= nn_cut)
    if reliable_negative_mask.sum() < 2_000:
        candidates = np.where(unlabeled_mask)[0]
        order = np.lexsort((nn_score[candidates], noisy_score[candidates]))
        reliable_negative_mask[candidates[order[: min(5_000, len(order))]]] = True

    pu_indices = np.where(known_mask | reliable_negative_mask)[0]
    pu_y = known_mask[pu_indices].astype(int)
    pu_models = fit_binary_ensemble(x_all[pu_indices], pu_y)
    pu_score = predict_binary_ensemble(pu_models, x_all)
    risk = 0.45 * pu_score + 0.35 * noisy_score + 0.20 * nn_score
    risk[known_mask] = 1.0

    y_family = label_map.loc[fqdn_ids[known_mask]].to_numpy(dtype=int)
    family_cv = family_oof_score(x_all[known_mask], y_family)
    classes = np.sort(np.unique(y_family))
    family_tree = ExtraTreesClassifier(
        n_estimators=520,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    family_logreg = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE)
    family_tree.fit(x_all[known_mask], y_family)
    family_logreg.fit(x_all[known_mask], y_family)
    family_proba = 0.58 * aligned_proba(family_tree, x_all, classes) + 0.42 * aligned_proba(family_logreg, x_all, classes)
    family_pred = classes[np.argmax(family_proba, axis=1)].astype(int)

    target_total = int(np.clip(round(len(labels) * 2.1), 900, 1100))
    min_total = 900
    max_total = 1100
    min_extra = max(0, min_total - int(known_mask.sum()))
    max_extra = max(0, max_total - int(known_mask.sum()))
    desired_extra = max(0, target_total - int(known_mask.sum()))
    unlabeled_indices = np.where(unlabeled_mask)[0]
    ranked_unlabeled = unlabeled_indices[np.argsort(-risk[unlabeled_indices])]
    # Keep the dns1 submission size deterministic.  The assignment states that
    # known labels cover roughly half of the malicious domains, so selecting the
    # top risk-ranked unlabeled domains up to the inferred target total is both
    # conservative and reproducible across runs.
    extra_count = int(np.clip(desired_extra, min_extra, max_extra))
    selected_extra = ranked_unlabeled[:extra_count]

    known_out = labels[["fqdn_no", "family_no"]].copy()
    extra_out = pd.DataFrame(
        {
            "fqdn_no": fqdn_ids[selected_extra],
            "family_no": family_pred[selected_extra].astype(int),
            "risk": risk[selected_extra],
        }
    ).sort_values("risk", ascending=False)
    out = pd.concat([known_out, extra_out[["fqdn_no", "family_no"]]], ignore_index=True)
    out = out.drop_duplicates("fqdn_no", keep="first")
    out["family_no"] = out["family_no"].astype(int)
    out.to_csv(RESULTS_DIR / "dns1" / "label.csv", index=False)

    selected_risk_min = float(risk[selected_extra].min()) if len(selected_extra) else 0.0
    selected_risk_mean = float(risk[selected_extra].mean()) if len(selected_extra) else 0.0
    summary.append(
        f"## dns1\n- Known malicious labels included: {len(labels):,}\n- Additional PU discoveries: {len(selected_extra):,}\n"
        f"- Submitted rows: {len(out):,}\n- Dense features: {x_all.shape[1]:,} "
        f"({len(feature_names):,} named numeric/SVD columns)\n- Reliable negatives used: {int(reliable_negative_mask.sum()):,}\n"
        f"- Family macro-F1 CV on known labels: {family_cv:.4f}\n- Extra risk mean/min: {selected_risk_mean:.4f}/{selected_risk_min:.4f}\n"
    )
    log(f"dns1 wrote {len(out):,} malicious-domain rows ({len(selected_extra):,} newly discovered)")


def solve_dns3(summary: list[str]) -> None:
    log("Starting dns3 unsupervised clustering")
    base = DATA_DIR / "dns3" / "question"
    fqdn = pd.read_csv(base / "fqdn.csv")
    fqdn["fqdn_no"] = fqdn["fqdn_no"].astype(str)
    access = aggregate_access(
        base / "access.csv",
        request_col="request_cnt",
        ip_col="encoded_ip",
        total_request_col="total_request",
        date_col="date",
        hour_col="hour",
        prefix="dns3_access",
    )
    flint, _ = aggregate_flint(base / "flint.csv", ttl_col="ttl", prefix="dns3_flint")
    features = join_feature_blocks(fqdn, [fqdn_lexical_features(fqdn), access, flint])
    x_all, feature_names = build_dense_matrix(fqdn, features, text_components=112, max_text_features=8500)

    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(3000, len(fqdn))
    sample_idx = np.sort(rng.choice(len(fqdn), size=sample_size, replace=False))
    candidates = [3, 4, 5, 6, 8, 10, 12]
    best_k = 3
    best_silhouette = -1.0
    best_labels: NDArray | None = None
    for k in candidates:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=2048,
            n_init=12,
            max_iter=220,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(x_all)
        unique = np.unique(labels)
        if len(unique) < 3:
            continue
        score = float(silhouette_score(x_all[sample_idx], labels[sample_idx], metric="euclidean"))
        log(f"  dns3 k={k} silhouette={score:.4f}")
        if score > best_silhouette:
            best_silhouette = score
            best_k = k
            best_labels = labels
    if best_labels is None:
        model = MiniBatchKMeans(n_clusters=3, random_state=RANDOM_STATE, batch_size=2048, n_init=12, max_iter=220)
        best_labels = model.fit_predict(x_all)
        best_k = len(np.unique(best_labels))
        best_silhouette = 0.0

    _, contiguous = np.unique(best_labels, return_inverse=True)
    out = pd.DataFrame({"fqdn_no": fqdn["fqdn_no"], "label": contiguous.astype(int)})
    out.to_csv(RESULTS_DIR / "dns3" / "label.csv", index=False)
    counts = out["label"].value_counts().sort_index().to_dict()
    summary.append(
        f"## dns3\n- Rows clustered: {len(out):,}\n- Clusters selected: {best_k}\n"
        f"- Dense features: {x_all.shape[1]:,} ({len(feature_names):,} named numeric/SVD columns)\n"
        f"- Sample silhouette: {best_silhouette:.4f}\n- Cluster sizes: {counts}\n"
    )
    log(f"dns3 wrote {len(out):,} cluster labels across {out['label'].nunique()} clusters")


def write_summary(sections: list[str]) -> None:
    text = "# DNS Homework Solver Run Summary\n\n"
    text += f"Generated by `scripts/solve_dns_homework.py` with random_state={RANDOM_STATE}.\n\n"
    text += "The solver uses encoded-FQDN lexical/character features, chunked access and Flint aggregates, "
    text += "shared DNS target graph features, and task-specific WHOIS/IP/TTL features where available.\n\n"
    text += "\n".join(sections)
    (REPORTS_DIR / "run_summary.md").write_text(text, encoding="utf-8")
    log(f"Wrote {REPORTS_DIR / 'run_summary.md'}")


def main() -> None:
    ensure_output_dirs()
    summary: list[str] = []
    solve_dns1(summary)
    solve_dns2(summary)
    solve_dns3(summary)
    write_summary(summary)
    log("All DNS homework outputs generated successfully")


if __name__ == "__main__":
    main()
