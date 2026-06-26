"""Single seam for assembling a jurisdiction's central-government tax config.

This module is the one place that builds a fully-populated
:class:`~macromodel.configurations.central_government_configuration.CentralGovernmentConfiguration`
for a real run.  It combines the two sources of tax inputs the project adds:

1. **Schedules** (multi-row, year-ranged tables) read from CSV files under
   ``spoof_data/freda/`` via ``PITSchedule`` / ``TaxCreditSchedule`` /
   ``DividendTaxCreditSchedule`` — the progressive bracket schedule, the
   non-refundable tax-credit amounts, and the dividend gross-up / DTC rates.
2. **Scalars** (single-value assumptions) read from ``tax_parameters.yaml`` via
   :func:`apply_tax_parameters` — the dividend-integration switch, the
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

Dividend integration activates on schedule presence
---------------------------------------------------
Within this opt-in path, dividend integration (the Canadian gross-up + dividend
tax credit) is switched on automatically when a dividend tax credit schedule
CSV is present: the builder loads its rates and sets
``pit_dividend_integration`` to ``True``.  When the schedule is absent the
builder leaves integration off and firm dividends keep the legacy treatment.
Schedule presence overrides the ``pit_dividend_integration`` switch in
``tax_parameters.yaml``; that YAML value governs only when no schedule is found.

Bracket units
-------------
``pit_brackets`` are returned in *per-individual* dollar units.  The scale to
agent-level units (each synthetic agent represents ``scale`` people) is applied
later in ``Country.from_pickled_country``; the builder must not pre-scale.

Tax-credit coverage
-------------------
The runtime credit pool (``pit_pools._credit_amount``) dispatches by credit
``kind`` and already implements universal, age-based, couple (Spousal Amount,
income-tested) and single-parent (Equivalent To Spouse Amount) credits, deriving
the household context per individual.  This builder therefore carries those
kinds.  Credits whose eligibility the runtime does not yet evaluate (the
per-child amounts, which require child-count context that is not always present)
are *skipped* and logged rather than silently applied to everyone.  The
allow-list below must stay in step with the ``kind`` branches in
``_credit_amount``: a key is admitted here only once the runtime can apply the
credit it gates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from macromodel.configurations.central_government_configuration import (
    CentralGovernmentConfiguration,
    TaxCreditDef,
)

from .tax_parameters_reader import apply_tax_parameters

if TYPE_CHECKING:
    from macro_data.readers.taxation.personal_income_tax.tax_credit_schedule import (
        TaxCreditComponent,
    )

logger = logging.getLogger(__name__)

# Eligibility keys the runtime credit pool can act on.  A credit whose
# eligibility dict contains any other key is not yet applicable at runtime and is
# skipped.  These map to the ``kind`` branches in ``pit_pools._credit_amount``:
#   age_min            -> Age Amount (and other age-gated credits)
#   in_couple_household -> Spousal Amount (income-tested against the spouse)
#   is_single_parent   -> Equivalent To Spouse Amount
# The per-child keys (num_children_under_*) are intentionally absent: the runtime
# branch exists but requires child-count context, so those remain deferred.
_EXPRESSIBLE_ELIGIBILITY_KEYS = frozenset(
    {"age_min", "in_couple_household", "is_single_parent"}
)


def _credit_component_to_def(component: TaxCreditComponent) -> Optional[TaxCreditDef]:
    """Map a data-layer ``TaxCreditComponent`` to a config-layer ``TaxCreditDef``.

    Returns ``None`` (and logs) when the component's eligibility uses rules the
    runtime credit pool cannot yet act on, so the caller can skip it instead of
    applying it universally.
    """
    extra_keys = set(component.eligibility) - _EXPRESSIBLE_ELIGIBILITY_KEYS
    if extra_keys:
        logger.warning(
            "Skipping tax credit '%s': eligibility rule(s) %s are not yet "
            "applied by the runtime credit pool (supported: universal, age, "
            "couple, single-parent). The credit is omitted to avoid applying it "
            "universally.",
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
    dividend_schedule_filename: Optional[str] = None,
    params_path: str | Path | None = None,
    base_config: Optional[CentralGovernmentConfiguration] = None,
) -> CentralGovernmentConfiguration:
    """Assemble a central-government configuration from CSV schedules + YAML scalars.

    Args:
        jurisdiction: Jurisdiction key, used both for the default schedule
            filenames and to look up the scalar block in ``tax_parameters.yaml``.
        tax_year: Tax year for which to compute the (CPI-indexed) brackets and
            credit amounts, select the dividend rates, and key the scalar lookup.
        schedule_filename: PIT bracket CSV under ``spoof_data/freda/``.  Defaults
            to ``f"{jurisdiction}_pit_2014.csv"`` (the base-year file; later years
            are reached by CPI-indexing that file).  The companion
            ``*_tax_credit_amount_*.csv`` is auto-discovered by ``PITSchedule``.
        dividend_schedule_filename: Dividend gross-up / DTC rate CSV under
            ``spoof_data/freda/``.  Defaults to
            ``f"{jurisdiction}_dividend_tax_credit_schedule.csv"``.  The row
            covering *tax_year* supplies the per-type gross-up and DTC rates.
        params_path: Optional override for the scalar YAML file location.
        base_config: Optional base configuration whose non-tax fields (functions,
            social benefits) are preserved.  Defaults to a fresh
            ``CentralGovernmentConfiguration()``.

    Returns:
        A ``CentralGovernmentConfiguration`` with ``pit_brackets``,
        ``pit_tax_credits`` and the dividend gross-up / DTC rates populated from
        the schedules, and the remaining scalar fields overridden from the YAML.
    """
    # Deferred imports: the data-layer schedule readers (and pandas) load only
    # when a config is actually built, so importing this package for the config
    # schema alone stays light.  This keeps the configurations layer free of an
    # import-time dependency on macro_data, matching the runtime-deferral pattern
    # used elsewhere (e.g. pit_pools._household_context, Firms.from_configuration).
    from macro_data.readers.taxation.personal_income_tax.dividend_tax_credit_schedule import (
        DividendTaxCreditSchedule,
    )
    from macro_data.readers.taxation.personal_income_tax.pit_schedule import (
        PITSchedule,
    )

    if schedule_filename is None:
        schedule_filename = f"{jurisdiction}_pit_2014.csv"
    if dividend_schedule_filename is None:
        dividend_schedule_filename = f"{jurisdiction}_dividend_tax_credit_schedule.csv"

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

    # ── Load the dividend gross-up / DTC rates for the year, if a schedule
    #    is present.  Presence of the schedule is the activation signal: when
    #    the file exists, dividend integration is switched on automatically and
    #    its rates are populated; when it is absent, integration stays off and
    #    firm dividends keep the legacy (no gross-up / no DTC) treatment.
    dividend_updates: dict = {}
    dividend_schedule_present = False
    try:
        dividend_schedule = DividendTaxCreditSchedule.from_name(
            dividend_schedule_filename
        )
    except FileNotFoundError:
        logger.info(
            "No dividend tax credit schedule '%s' found under spoof_data/freda/; "
            "dividend integration stays off and firm dividends keep the legacy "
            "treatment.",
            dividend_schedule_filename,
        )
    else:
        dividend_schedule_present = True
        dividend_rates = dividend_schedule.get_year_rates(tax_year=tax_year)
        eligible = dividend_rates["eligible"]
        non_eligible = dividend_rates["non_eligible"]
        dividend_updates = {
            "dividend_eligible_gross_up": eligible.gross_up_rate,
            "dividend_non_eligible_gross_up": non_eligible.gross_up_rate,
            "dividend_eligible_dtc_rate": eligible.dtc_rate_of_grossed_up,
            "dividend_non_eligible_dtc_rate": non_eligible.dtc_rate_of_grossed_up,
        }

    # ── Apply schedules, then the scalar overrides from the YAML ──
    config = base.model_copy(
        update={
            "pit_brackets": pit_brackets,
            "pit_tax_credits": pit_tax_credits,
            **dividend_updates,
        }
    )
    config = apply_tax_parameters(
        config, jurisdiction=jurisdiction, year=tax_year, path=params_path
    )
    # The dividend schedule's presence activates integration.  This is applied
    # AFTER the YAML scalars so schedule-presence wins over the YAML switch (the
    # YAML value governs only when no schedule is present).
    if dividend_schedule_present:
        config = config.model_copy(update={"pit_dividend_integration": True})
    return config
