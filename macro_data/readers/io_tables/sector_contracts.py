"""Sector-system bridges used before model data are initialized.

The helpers here intentionally avoid broad prefix fallbacks.  They keep exact
matches where possible and only split/merge source sectors through explicit
rules for known differences between WIOD/SEA and the IO sector list currently
being read.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


SEA_TO_IO_CANDIDATE_TARGETS: dict[str, list[str]] = {
    # WIOD SEA has aggregate mining B.  The current provincial IO table splits
    # oil/gas/coal and keeps B07/B09, while older tables may keep B05.
    "B": ["B05a", "B05b", "B05c", "B05", "B07", "B09"],
    "B05": ["B05a", "B05b", "B05c"],
    # Current provincial IO table collapses agriculture and information sectors.
    "A01": ["A"],
    "A02": ["A"],
    "A03": ["A"],
    "J58T60": ["J"],
    "J61": ["J"],
    "J62": ["J"],
    # WIOD naming variants before current IO names.
    "J58": ["J58T60", "J"],
    "J59_J60": ["J58T60", "J"],
    "J62_J63": ["J62", "J"],
    # Metal split used by the current provincial IO table.
    "C24": ["C24a", "C24b"],
    # Older 50-sector tables split electricity. Current 43-sector tables keep D.
    "D": ["D01a", "D01b", "D01c", "D01d", "D01e"],
    # Current IO sector list keeps the broad residual service bucket.
    "T": ["R_S"],
    "U": ["R_S"],
}


SEA_AGGREGATE_PREFIXES = {
    "A",
    "C",
    "D",
    "E",
    "G",
    "H",
    "J",
    "K",
    "M",
}


def bridge_sea_to_io_industries(
    sea: pd.DataFrame,
    industries: Iterable[str],
    value_added_dict: dict[str, pd.Series],
    country_names: Iterable[str],
) -> pd.DataFrame:
    """Return SEA data aligned to the actual IO industry list.

    Args:
        sea: Data indexed by ``(country, source_industry)`` with SEA variables
            as columns.
        industries: The target IO/model industry list read from the IO table.
        value_added_dict: IO value added by target industry, used as split
            weights when one source sector maps to multiple target sectors.
        country_names: Countries present before optional provincial splitting.

    Raises:
        ValueError: If a source SEA industry cannot be mapped explicitly.
    """

    target_industries = list(industries)
    target_set = set(target_industries)
    countries = list(country_names)

    exact = sea.loc[sea.index.get_level_values(1).isin(target_set)]
    pieces = [exact] if not exact.empty else []
    unmapped: list[str] = []

    source_industries = sea.index.get_level_values(1).unique()
    for source_industry in source_industries:
        if source_industry in target_set:
            continue

        targets = _resolve_sea_targets(source_industry, target_industries)
        if not targets:
            unmapped.append(source_industry)
            continue

        for country in countries:
            if (country, source_industry) not in sea.index:
                continue
            source_values = sea.loc[(country, source_industry)]
            weights = _target_value_added_weights(value_added_dict[country], targets)
            split = pd.DataFrame(
                [source_values.values * weight for weight in weights],
                index=pd.MultiIndex.from_product([[country], targets], names=sea.index.names),
                columns=sea.columns,
            )
            pieces.append(split)

    if unmapped:
        raise ValueError(
            "Cannot align WIOD SEA sectors to IO industries without explicit rules: "
            + ", ".join(sorted(set(unmapped)))
        )

    if not pieces:
        return pd.DataFrame(
            0.0,
            index=pd.MultiIndex.from_product([countries, target_industries], names=sea.index.names),
            columns=sea.columns,
        )

    aligned = pd.concat(pieces).groupby(level=[0, 1]).sum()
    complete_index = pd.MultiIndex.from_product([countries, target_industries], names=sea.index.names)
    return aligned.reindex(complete_index, fill_value=0.0)


def _resolve_sea_targets(source_industry: str, target_industries: list[str]) -> list[str]:
    target_set = set(target_industries)

    explicit_targets = [
        target for target in SEA_TO_IO_CANDIDATE_TARGETS.get(source_industry, []) if target in target_set
    ]
    if explicit_targets:
        return explicit_targets

    if source_industry in SEA_AGGREGATE_PREFIXES:
        prefix_targets = [target for target in target_industries if target.startswith(source_industry)]
        if prefix_targets:
            return prefix_targets

    return []


def _target_value_added_weights(value_added: pd.Series, targets: list[str]) -> np.ndarray:
    weights = value_added.reindex(targets).fillna(0.0).to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if weights.sum() == 0.0:
        return np.repeat(1.0 / len(targets), len(targets))
    return weights / weights.sum()
