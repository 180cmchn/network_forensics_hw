"""DNS3 feature construction helpers for score experiments."""

from __future__ import annotations

from typing import Any

try:
    import solve_dns_homework as solver
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import solve_dns_homework as solver

try:
    from score_exp_common import matrix_shape
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_common import matrix_shape

try:
    from score_exp_types import Dns3FeatureBundle
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_types import Dns3FeatureBundle


def build_dns3_feature_bundle() -> Dns3FeatureBundle:
    pd = solver.pd
    np = solver.np
    base = solver.DATA_DIR / "dns3" / "question"
    fqdn = pd.read_csv(base / "fqdn.csv")
    fqdn["fqdn_no"] = fqdn["fqdn_no"].astype(str)
    access = solver.aggregate_access(
        base / "access.csv",
        request_col="request_cnt",
        ip_col="encoded_ip",
        total_request_col="total_request",
        date_col="date",
        hour_col="hour",
        prefix="dns3_access",
    )
    flint, _ = solver.aggregate_flint(base / "flint.csv", ttl_col="ttl", prefix="dns3_flint")
    features = solver.join_feature_blocks(fqdn, [solver.fqdn_lexical_features(fqdn), access, flint])
    matrix, feature_names = solver.build_dense_matrix(fqdn, features, text_components=112, max_text_features=8500)
    rng = np.random.default_rng(solver.RANDOM_STATE)
    sample_size = min(3000, len(fqdn))
    sample_indices = np.sort(rng.choice(len(fqdn), size=sample_size, replace=False))
    text_column_indices = [index for index, name in enumerate(feature_names) if name.startswith("char_svd_")]
    return Dns3FeatureBundle(
        fqdn=fqdn,
        matrix=matrix,
        feature_names=feature_names,
        sample_indices=sample_indices,
        text_column_indices=text_column_indices,
    )


def reduced_dns3_matrix(matrix: Any, *, components: int = 48) -> Any:
    np = solver.np
    n_components = min(components, max(2, matrix.shape[1] - 1), max(2, matrix.shape[0] - 1))
    if n_components >= matrix.shape[1]:
        return matrix.astype(np.float32, copy=False)
    reducer = solver.TruncatedSVD(n_components=n_components, random_state=solver.RANDOM_STATE)
    reduced = reducer.fit_transform(matrix)
    return solver.StandardScaler().fit_transform(reduced).astype(np.float32)


def text_weighted_dns3_matrix(bundle: Dns3FeatureBundle, *, text_weight: float = 0.5) -> Any:
    weighted = bundle.matrix.copy()
    if bundle.text_column_indices:
        weighted[:, bundle.text_column_indices] *= text_weight
    return weighted


def dns3_feature_indices(bundle: Dns3FeatureBundle, prefixes: tuple[str, ...]) -> list[int]:
    return [index for index, name in enumerate(bundle.feature_names) if name.startswith(prefixes)]


def dns3_matrix_with_columns(bundle: Dns3FeatureBundle, indices: list[int]) -> Any:
    if not indices:
        raise ValueError("dns3 focus feature subset is empty")
    return bundle.matrix[:, indices].astype(solver.np.float32, copy=False)


def dns3_scaled_numeric_block(bundle: Dns3FeatureBundle, features: Any) -> tuple[Any, list[str]]:
    pd = solver.pd
    np = solver.np
    if features is None or features.empty:
        return np.empty((len(bundle.fqdn), 0), dtype=np.float32), []
    fqdn_order = bundle.fqdn["fqdn_no"].astype(str).tolist()
    numeric = features.reindex(fqdn_order).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not numeric.empty:
        variances = numeric.var(axis=0)
        numeric = numeric.loc[:, variances > 1e-12]
    if numeric.empty:
        return np.empty((len(bundle.fqdn), 0), dtype=np.float32), []
    scaled = solver.StandardScaler().fit_transform(numeric.to_numpy(dtype=np.float64)).astype(np.float32)
    return scaled, numeric.columns.tolist()


def dns3_ip_metadata_matrix(bundle: Dns3FeatureBundle) -> tuple[Any, list[str], dict[str, Any]]:
    pd = solver.pd
    base = solver.DATA_DIR / "dns3" / "question"
    _, pairs = solver.aggregate_flint(base / "flint.csv", ttl_col="ttl", prefix="dns3_flint", return_pairs=True)
    ipmeta = solver.aggregate_ip_metadata(
        [base / "ip.csv", base / "ipv6.csv"],
        pairs if pairs is not None else pd.DataFrame(),
        prefix="dns3_ipmeta",
    )
    matrix, feature_names = dns3_scaled_numeric_block(bundle, ipmeta)
    return matrix, feature_names, {
        "ip_metadata_source_files": ["data/dns3/question/ip.csv", "data/dns3/question/ipv6.csv"],
        "ip_metadata_raw_feature_count": int(0 if ipmeta is None else len(ipmeta.columns)),
        "ip_metadata_selected_feature_count": int(len(feature_names)),
        "ip_metadata_available": bool(feature_names),
        "ip_metadata_fill_zero": True,
    }


def build_dns3_interpretable_routes(bundle: Dns3FeatureBundle) -> dict[str, tuple[Any, dict[str, Any]]]:
    all_indices = list(range(len(bundle.feature_names)))
    access_indices = set(dns3_feature_indices(bundle, ("dns3_access",)))
    flint_indices = set(dns3_feature_indices(bundle, ("dns3_flint",)))
    lexical_indices = set(dns3_feature_indices(bundle, ("lex_",)))
    text_indices = set(bundle.text_column_indices)
    traffic_indices = access_indices | flint_indices
    ordered_traffic_indices = [index for index in all_indices if index in traffic_indices]
    ordered_traffic_lexical_indices = [index for index in all_indices if index in traffic_indices or index in lexical_indices or index in text_indices]

    traffic_matrix = dns3_matrix_with_columns(bundle, ordered_traffic_indices)
    traffic_feature_names = [bundle.feature_names[index] for index in ordered_traffic_indices]
    ip_matrix, ip_feature_names, ip_params = dns3_ip_metadata_matrix(bundle)
    if ip_feature_names:
        traffic_plus_ip_matrix = solver.np.hstack([traffic_matrix, ip_matrix]).astype(solver.np.float32, copy=False)
    else:
        traffic_plus_ip_matrix = traffic_matrix.copy()
    traffic_plus_ip_feature_names = traffic_feature_names + ip_feature_names

    route_specs = {
        "traffic_only": (traffic_matrix, traffic_feature_names, {}),
        "traffic_plus_ip": (traffic_plus_ip_matrix, traffic_plus_ip_feature_names, ip_params),
        "traffic_plus_lexical": (
            dns3_matrix_with_columns(bundle, ordered_traffic_lexical_indices),
            [bundle.feature_names[index] for index in ordered_traffic_lexical_indices],
            {},
        ),
    }
    routes: dict[str, tuple[Any, dict[str, Any]]] = {}
    for route_name, (matrix, feature_names, extra_params) in route_specs.items():
        routes[route_name] = (
            matrix,
            {
                "route": route_name,
                "matrix": route_name,
                "matrix_shape": matrix_shape(matrix),
                "selected_feature_count": int(len(feature_names)),
                "traffic_feature_count": int(len(traffic_feature_names)),
                "access_feature_count": int(len(access_indices)),
                "flint_feature_count": int(len(flint_indices)),
                "lexical_feature_count": int(sum(name.startswith("lex_") for name in feature_names)),
                "char_svd_feature_count": int(sum(name.startswith("char_svd_") for name in feature_names)),
                "ip_metadata_feature_count": int(sum(name.startswith("dns3_ipmeta") for name in feature_names)),
                "feature_names": feature_names,
                **extra_params,
            },
        )
    return routes


__all__ = [
    "build_dns3_feature_bundle",
    "reduced_dns3_matrix",
    "text_weighted_dns3_matrix",
    "dns3_feature_indices",
    "dns3_matrix_with_columns",
    "dns3_scaled_numeric_block",
    "dns3_ip_metadata_matrix",
    "build_dns3_interpretable_routes",
]
