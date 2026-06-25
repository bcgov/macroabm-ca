"""Single seam for assembling a jurisdiction's central-government tax config.

This module is the one place that builds a fully-populated
:class:`~macromodel.configurations.central_government_configuration.CentralGovernmentConfiguration`
for a real run.  It combines the two sources of tax inputs the project adds:

1. **Schedules** (multi-row, CPI-indexed tables) read from CSV files under
   ``spoof_data/freda/`` via ``PITSchedule`` / ``TaxCreditSchedule`` — the
   progressive bracket schedule and the non-refundable tax-credit amounts.
2. **Scalars** (single-value assumptions) read from ``tax_parameters.yaml`` via
   :func:`apply_tax_parameters` — dividend gross-up / DTC rates, the
   small-business share, and the couple rental-income split.

The result is a configuration object the existing ``Country.from_pickled_country``
flow consumes unchanged: it reads ``pit_brackets`` (and scales the thresholds to
agent units), ``pit_tax_credits``, and the scalar fields.

Opt-in by design
----------------
Calling this builder activates the progressive PIT schedule, which changes
government revenue relative to the upstream flat-rate behaviour.  It is
therefore *opt-in*: the default ``CountryConfiguration()`` stays flat, and a
scenario must explicitly call this builder to switch BC taxation on.

Bracket units
-------------
``pit_brackets`` are returned in *per-individual* dollar units.  The scale to
agent-level units (each synthetic agent represents ``scale`` people) is applied
later in ``Country.from_pickled_country``; the builder must not pre-scale.

Tax-credit coverage
-------------------
The configuration layer (``TaxCreditDef``) can currently express only universal
and age-based eligibility (plus clawback bounds).  Credits whose eligibility
depends on household composition (e.g. Spousal Amount, Equivalent To Spouse) are
*skipped* and logged, rather than silently applied to everyone — applying them
universally would understate revenue.  They become includable once their
eligibility is plumbed through ``TaxCreditDef`` and the runtime credit state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from macro_data.readers.taxation.personal_income_tax.pit_schedule import PITSchedule
from macro_data.readers.taxation.personal_income_tax.tax_credit_schedule import (
    TaxCreditComponent,
)
from macromodel.configurations.central_government_configuration import (
    CentralGovernmentConfiguration,
    TaxCreditDef,
)

from .tax_parameters_reader import apply_tax_parameters

logger = logging.getLogger(__name__)

# Eligibility keys that TaxCreditDef can represent.  A credit whose eligibility
# dict contains any other key cannot be expressed yet and is skipped.
_EXPRESSIBLE_ELIGIBILITY_KEYS = frozenset({"age_min"})


def _credit_component_to_def(component: TaxCreditComponent) -> Optional[TaxCreditDef]:
    """Map a data-layer ``TaxCreditComponent`` to a config-layer ``TaxCreditDef``.

    Returns ``None`` (and logs) when the component's eligibility uses rules the
    configuration layer cannot yet express, so the caller can skip it instead of
    applying it universally.
    """
    extra_keys = set(component.eligibility) - _EXPRESSIBLE_ELIGIBILITY_KEYS
    if extra_keys:
        logger.warning(
            "Skipping tax credit '%s': eligibility rule(s) %s are not yet "
            "expressible in TaxCreditDef (only universal/age-based credits are "
            "supported). The credit is omitted to avoid applying it universally.",
            component.kind,
            sorted(extra_keys),
        )
        return None

    return TaxCreditDef(
        kind=component.kind,
        amount=component.amount,
        indexing=component.indexing,
        eligibility_age_min=component.eligibility.get("age_min"),
        clawback_start=component.clawback_start,
        clawback_cap=component.clawback_cap,
    )


def build_central_government_configuration(
    jurisdiction: str = "bc",
    tax_year: int = 2014,
    schedule_filename: Optional[str] = None,
    params_path: str | Path | None = None,
    base_config: Optional[CentralGovernmentConfiguration] = None,
) -> CentralGovernmentConfiguration:
    """Assemble a central-government configuration from CSV schedules + YAML scalars.

    Args:
        jurisdiction: Jurisdiction key, used both for the default schedule
            filename and to look up the scalar block in ``tax_parameters.yaml``.
        tax_year: Tax year for which to compute the (CPI-indexed) brackets and
            credit amounts, and the year key for the scalar lookup.
        schedule_filename: PIT bracket CSV under ``spoof_data/freda/``.  Defaults
            to ``f"{jurisdiction}_pit_2014.csv"`` (the base-year file; later years
            are reached by CPI-indexing that file).  The companion
            ``*_tax_credit_amount_*.csv`` is auto-discovered by ``PITSchedule``.
        params_path: Optional override for the scalar YAML file location.
        base_config: Optional base configuration whose non-tax fields (functions,
            social benefits) are preserved.  Defaults to a fresh
            ``CentralGovernmentConfiguration()``.

    Returns:
        A ``CentralGovernmentConfiguration`` with ``pit_brackets`` and
        ``pit_tax_credits`` populated from the schedules and the scalar fields
        overridden from the YAML.
    """
    if schedule_filename is None:
        schedule_filename = f"{jurisdiction}_pit_2014.csv"

    base = base_config if base_config is not None else CentralGovernmentConfiguration()

    # ── Load the bracket schedule (and its companion credit schedule) ──
    # For the base year no CPI is needed; later years require the CPI map to
    # compound-inflate indexed bounds, so use the CPI-aware factory then.
    schedule = PITSchedule.from_name(schedule_filename)
    if tax_year > schedule.base_year:
        schedule = PITSchedule.from_name_with_cpi(schedule_filename)

    thresholds, rates, _, _ = schedule.get_brackets(tax_year=tax_year)
    pit_brackets = [(float(t), float(r)) for t, r in zip(thresholds, rates)]

    # ── Map the companion tax credits, skipping the not-yet-expressible ones ──
    pit_tax_credits: Optional[list[TaxCreditDef]] = None
    if schedule.tax_credits is not None:
        components = schedule.tax_credits.get_credits(tax_year=tax_year)
        mapped = [
            d for d in (_credit_component_to_def(c) for c in components) if d is not None
        ]
        pit_tax_credits = mapped or None

    # ── Apply schedules, then the scalar overrides from the YAML ──
    config = base.model_copy(
        update={"pit_brackets": pit_brackets, "pit_tax_credits": pit_tax_credits}
    )
    config = apply_tax_parameters(
        config, jurisdiction=jurisdiction, year=tax_year, path=params_path
    )
    return config
