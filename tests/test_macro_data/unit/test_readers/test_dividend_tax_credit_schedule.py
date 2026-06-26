"""Unit tests for the dividend gross-up / DTC rate schedule reader."""

import tempfile
from pathlib import Path

import pytest

from macro_data.readers.taxation.personal_income_tax.dividend_tax_credit_schedule import (
    DividendRates,
    DividendTaxCreditSchedule,
)


# ═══════════════════════════════════════════════════════════════════════
# from_name — the packaged BC schedule
# ═══════════════════════════════════════════════════════════════════════


class TestPackagedSchedule:
    def test_2014_eligible_rates(self):
        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        rates = schedule.get_rates(2014, "eligible")
        assert rates.gross_up_rate == pytest.approx(0.38)
        assert rates.dtc_rate_of_grossed_up == pytest.approx(0.10)
        assert rates.dtc_rate_of_actual == pytest.approx(0.138)

    def test_2014_non_eligible_rates(self):
        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        rates = schedule.get_rates(2014, "non_eligible")
        assert rates.gross_up_rate == pytest.approx(0.18)
        assert rates.dtc_rate_of_grossed_up == pytest.approx(0.0259)
        assert rates.dtc_rate_of_actual == pytest.approx(0.0306)

    def test_get_year_rates_returns_both_types(self):
        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        year = schedule.get_year_rates(2014)
        assert set(year) == {"eligible", "non_eligible"}
        assert year["eligible"].dtc_rate_of_grossed_up == pytest.approx(0.10)
        assert year["non_eligible"].gross_up_rate == pytest.approx(0.18)

    def test_open_ended_range_applies_to_later_year(self):
        """The 2019-onward row (blank year_to) covers a later year like 2020."""
        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        eligible = schedule.get_rates(2020, "eligible")
        non_eligible = schedule.get_rates(2020, "non_eligible")
        assert eligible.dtc_rate_of_grossed_up == pytest.approx(0.12)
        assert non_eligible.gross_up_rate == pytest.approx(0.15)

    def test_multi_year_range_applies_midrange(self):
        """The eligible 2012-2018 row covers an interior year like 2016."""
        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        rates = schedule.get_rates(2016, "eligible")
        assert rates.gross_up_rate == pytest.approx(0.38)
        assert rates.dtc_rate_of_grossed_up == pytest.approx(0.10)

    def test_year_before_first_row_raises(self):
        """Eligible rows start in 2009; 2008 has no applicable row."""
        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        with pytest.raises(ValueError, match="covers tax year 2008"):
            schedule.get_rates(2008, "eligible")

    def test_unknown_dividend_type_raises(self):
        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        with pytest.raises(ValueError, match="No rows for dividend_type"):
            schedule.get_rates(2014, "preferred")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Dividend-rate file not found"):
            DividendTaxCreditSchedule.from_name("does_not_exist.csv")


# ═══════════════════════════════════════════════════════════════════════
# from_csv — synthetic CSVs
# ═══════════════════════════════════════════════════════════════════════


def _write_csv(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        return f.name


class TestFromCsv:
    def test_missing_required_column_raises(self):
        csv = "dividend_type,year_from,gross_up_rate\neligible,2014,0.38\n"
        p = _write_csv(csv)
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                DividendTaxCreditSchedule.from_csv(p)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_overlapping_rows_raise(self):
        csv = (
            "dividend_type,year_from,year_to,gross_up_rate,bc_dtc_pct_of_grossed_up\n"
            "eligible,2010,2015,0.38,0.10\n"
            "eligible,2014,2018,0.38,0.11\n"  # overlaps 2014-2015
        )
        p = _write_csv(csv)
        try:
            schedule = DividendTaxCreditSchedule.from_csv(p)
            with pytest.raises(ValueError, match="Overlapping"):
                schedule.get_rates(2014, "eligible")
        finally:
            Path(p).unlink(missing_ok=True)

    def test_actual_rate_optional(self):
        """A CSV without bc_dtc_pct_of_actual yields None for that field."""
        csv = (
            "dividend_type,year_from,year_to,gross_up_rate,bc_dtc_pct_of_grossed_up\n"
            "eligible,2014,,0.38,0.10\n"
        )
        p = _write_csv(csv)
        try:
            schedule = DividendTaxCreditSchedule.from_csv(p)
            rates = schedule.get_rates(2014, "eligible")
            assert rates == DividendRates("eligible", 0.38, 0.10, None)
        finally:
            Path(p).unlink(missing_ok=True)
