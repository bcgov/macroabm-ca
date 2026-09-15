"""Tests that the initial rental income tax is levied on private rent only.

Social-housing rent is government revenue, booked separately as `Total Social Housing Rent`.
Taxing it as well counts it twice on the GDP income side and breaks the construction-time
output == income identity by exactly `income_tax x Total Social Housing Rent`.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from macro_data.processing.synthetic_central_government.default_synthetic_central_government import (
    DefaultSyntheticCGovernment,
)

SOCIAL_HOUSING_TENURE = -1
PRIVATE_RENTAL_TENURE = 3

INCOME_TAX_RATE = 0.09
SOCIAL_HOUSING_RENT_PER_HOUSEHOLD = 100.0
PRIVATE_RENT_PER_HOUSEHOLD = 300.0

# Two social-housing households and two private renters.
TOTAL_SOCIAL_HOUSING_RENT = 2 * SOCIAL_HOUSING_RENT_PER_HOUSEHOLD
TOTAL_RENT_PAID = TOTAL_SOCIAL_HOUSING_RENT + 2 * PRIVATE_RENT_PER_HOUSEHOLD


def build_government() -> DefaultSyntheticCGovernment:
    """Return a government whose data frame holds the single row `set_revenue` writes into."""
    return DefaultSyntheticCGovernment(
        country_name="TST",
        year=2014,
        central_gov_data=pd.DataFrame(index=[0]),
        other_benefits_model=None,
        unemployment_benefits_model=None,
    )


def build_collaborators() -> dict:
    """Return `update_fields` arguments for a population that does contain social housing.

    Every rate other than the income tax is zero so the assertions isolate the rental term.
    """
    household_data = pd.DataFrame(
        {
            "Tenure Status of the Main Residence": [
                SOCIAL_HOUSING_TENURE,
                SOCIAL_HOUSING_TENURE,
                PRIVATE_RENTAL_TENURE,
                PRIVATE_RENTAL_TENURE,
            ],
            "Rent Paid": [
                SOCIAL_HOUSING_RENT_PER_HOUSEHOLD,
                SOCIAL_HOUSING_RENT_PER_HOUSEHOLD,
                PRIVATE_RENT_PER_HOUSEHOLD,
                PRIVATE_RENT_PER_HOUSEHOLD,
            ],
            "Income from Financial Assets": [0.0, 0.0, 0.0, 0.0],
        }
    )
    industry_vectors = pd.DataFrame(
        {
            "Household Consumption in LCU": [0.0],
            "Exports in LCU": [0.0],
            "Household Capital Inputs in LCU": [0.0],
        }
    )
    return {
        "tax_data": SimpleNamespace(
            income_tax=INCOME_TAX_RATE,
            employee_social_insurance_tax=0.0,
            employer_social_insurance_tax=0.0,
            export_tax=0.0,
            value_added_tax=0.0,
            capital_formation_tax=0.0,
        ),
        "synthetic_population": SimpleNamespace(
            household_data=household_data,
            individual_data=pd.DataFrame({"Employee Income": [0.0]}),
            social_housing_rent=SOCIAL_HOUSING_RENT_PER_HOUSEHOLD,
        ),
        "synthetic_firms": SimpleNamespace(
            firm_data=pd.DataFrame({"Taxes paid on Production": [0.0], "Corporate Taxes Paid": [0.0]})
        ),
        "synthetic_banks": SimpleNamespace(bank_data=pd.DataFrame({"Corporate Taxes Paid": [0.0]})),
        "industry_data": {"industry_vectors": industry_vectors},
    }


class TestRentalIncomeTaxBase:
    def test__social_housing_rent_is_excluded_from_the_tax_base(self):
        """The rental tax is levied on private rent only, not on all rent paid."""
        government = build_government()
        government.update_fields(**build_collaborators())

        # Guard against a vacuous pass: without social-housing households the two formulas agree.
        booked_social_housing_rent = government.central_gov_data["Total Social Housing Rent"].iloc[0]
        assert booked_social_housing_rent == pytest.approx(TOTAL_SOCIAL_HOUSING_RENT)
        assert booked_social_housing_rent > 0

        expected = INCOME_TAX_RATE * (TOTAL_RENT_PAID - TOTAL_SOCIAL_HOUSING_RENT)
        assert government.central_gov_data["Rental Income Taxes"].iloc[0] == pytest.approx(expected)
