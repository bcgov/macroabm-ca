"""Bundled personal-income-tax schedules read from a taxation data directory.

``TaxationReader`` is the data-layer handle for a jurisdiction's personal
income tax schedules.  It is constructed by ``DataReaders.from_raw_data`` when a
``taxation`` tree is present under the raw-data root (optional and additive, like
the energy-sector readers), and consumed by
``build_central_government_configuration`` to assemble the central-government
configuration.

It loads the progressive bracket schedule (with its companion non-refundable
tax-credit schedule, auto-discovered by ``PITSchedule``) and, when present, the
dividend gross-up / DTC rate schedule.  The schedules are year-flexible: the
consumer selects a tax year when reading brackets, credits, and dividend rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from macro_data.readers.taxation.personal_income_tax.dividend_tax_credit_schedule import (
    DividendTaxCreditSchedule,
)
from macro_data.readers.taxation.personal_income_tax.pit_schedule import PITSchedule


@dataclass
class TaxationReader:
    """Loaded personal-income-tax schedules for one jurisdiction.

    Attributes:
        pit_schedule: Progressive bracket schedule (CPI-aware, so any tax year at
            or after its base year can be requested); carries the companion
            non-refundable tax-credit schedule via ``pit_schedule.tax_credits``.
        dividend_schedule: Dividend gross-up / DTC rate schedule, or ``None`` when
            no dividend schedule is present (then dividend integration stays off).
        jurisdiction: The taxing-authority key these schedules belong to (e.g.
            ``"bc"``).  Carried so a consumer can attach the schedules to the
            matching government agent — the basis for supporting multiple
            government agents (federal / provincial / ...) later.
    """

    pit_schedule: PITSchedule
    dividend_schedule: Optional[DividendTaxCreditSchedule]
    jurisdiction: str = "bc"

    @classmethod
    def from_dir(cls, schedule_dir: Path, *, jurisdiction: str = "bc") -> "TaxationReader":
        """Load the schedules for *jurisdiction* from *schedule_dir*.

        Args:
            schedule_dir: Directory holding the schedule CSVs — typically
                ``raw_data_path / "taxation" / "personal_income_tax"``.
            jurisdiction: Jurisdiction key selecting the schedule filenames
                (``f"{jurisdiction}_pit_2014.csv"`` and
                ``f"{jurisdiction}_dividend_tax_credit_schedule.csv"``).

        Returns:
            A ``TaxationReader`` with the bracket schedule (and its companion
            credits) loaded, and the dividend schedule loaded when present.
        """
        schedule_dir = Path(schedule_dir)
        pit_filename = f"{jurisdiction}_pit_2014.csv"
        dividend_filename = f"{jurisdiction}_dividend_tax_credit_schedule.csv"

        # Load CPI-aware so the schedule supports the base year and later years
        # (the CPI map is read from the directory's cache or the CSV's embedded
        # column).  The companion tax-credit schedule is auto-discovered.
        pit_schedule = PITSchedule.from_name_with_cpi(
            pit_filename, schedule_dir=schedule_dir
        )

        try:
            dividend_schedule: Optional[DividendTaxCreditSchedule] = (
                DividendTaxCreditSchedule.from_name(
                    dividend_filename, schedule_dir=schedule_dir
                )
            )
        except FileNotFoundError:
            dividend_schedule = None

        return cls(
            pit_schedule=pit_schedule,
            dividend_schedule=dividend_schedule,
            jurisdiction=jurisdiction,
        )
