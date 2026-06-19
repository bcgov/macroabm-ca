"""Central Government agent implementation for macroeconomic modeling.

This module implements the central government agent, which manages:
- Tax collection and administration
- Social benefits distribution
- Fiscal policy implementation
- Government debt management

The central government plays a crucial role in:
- Revenue generation through various tax instruments
- Social welfare through benefits and transfers
- Economic stabilization through fiscal policy
- Public finance management
"""

from typing import Any, Optional

import h5py
import numpy as np

from macro_data import SyntheticCentralGovernment
from macro_data.processing import TaxData
from macromodel.agents.agent import Agent
from macromodel.agents.central_government.central_government_ts import (
    create_central_government_timeseries,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions
from macro_data.readers.taxation.personal_income_tax.pit_schedule import compute_progressive_tax


class CentralGovernment(Agent):
    """Central Government agent responsible for fiscal policy and social benefits.

    This class implements government fiscal operations including:
    - Tax collection (VAT, income, corporate, etc.)
    - Social benefit distribution (unemployment, other transfers)
    - Public finance management (revenue, deficit, debt)

    The agent manages multiple tax instruments:
    - Value-added Tax (VAT)
    - Income Tax
    - Corporate Tax
    - Social Insurance Contributions
    - Export and Capital Formation Taxes

    Attributes:
        functions (dict[str, Any]): Mapping of function names to implementations
        states (dict[str, float | np.ndarray]): Current state variables including
            tax rates and benefit models
        ts (TimeSeries): Time series data for government variables
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        """Initialize the Central Government agent.

        Args:
            country_name (str): Name of the country this government represents
            all_country_names (list[str]): List of all countries in the model
            n_industries (int): Number of industries in the economy
            functions (dict[str, Any]): Function implementations for government operations
            ts (TimeSeries): Time series data for tracking variables
            states (dict[str, float | np.ndarray]): State variables including tax rates
        """
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            0,
            0,
            ts,
            states,
        )
        self.functions = functions

        # Snapshot base thresholds for CPI inflation indexing.
        # When step_pit_brackets() is called mid-simulation, the stored
        # nominal values are compound-inflated and written back to states.
        if "pit_thresholds" in states:
            self.pit_base_thresholds = states["pit_thresholds"].copy()
        else:
            self.pit_base_thresholds = None

        # Snapshot base taxable-income deduction for CPI inflation indexing.
        self.pit_base_deductions: Optional[float] = states.get("pit_taxable_income_deductions")

        # Snapshot base tax credits (list of dicts) for CPI indexing.
        self.pit_base_tax_credits: Optional[list[dict]] = states.get("pit_tax_credits")

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_central_government: SyntheticCentralGovernment,
        configuration: CentralGovernmentConfiguration,
        n_industries: int,
        country_name: str,
        all_country_names: list[str],
        tax_data: TaxData,
        number_of_unemployed_individuals: int,
        taxes_net_subsidies: np.ndarray,
    ):
        """Create a Central Government instance from pickled data.

        Initializes the government with:
        - Tax rates from historical data
        - Benefit models from synthetic data
        - Configuration parameters
        - Country-specific settings

        Args:
            synthetic_central_government (SyntheticCentralGovernment): Synthetic data
            configuration (CentralGovernmentConfiguration): Configuration parameters
            n_industries (int): Number of industries
            country_name (str): Country name
            all_country_names (list[str]): All country names
            tax_data (TaxData): Historical tax rate data
            number_of_unemployed_individuals (int): Count of unemployed
            taxes_net_subsidies (np.ndarray): Net tax rates by sector

        Returns:
            CentralGovernment: Initialized government agent
        """
        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.central_government")

        states = {
            "Value-added Tax": tax_data.value_added_tax,
            "Export Tax": tax_data.export_tax,
            "Employer Social Insurance Tax": tax_data.employer_social_insurance_tax,
            "Employee Social Insurance Tax": tax_data.employee_social_insurance_tax,
            "Profit Tax": tax_data.profit_tax,
            "Income Tax": tax_data.income_tax,
            "Capital Formation Tax": tax_data.capital_formation_tax,
            "Taxes Less Subsidies Rates": taxes_net_subsidies,
            "unemployment_benefits_model": synthetic_central_government.unemployment_benefits_model,
            "other_benefits_model": synthetic_central_government.other_benefits_model,
        }

        # Progressive PIT schedule (optional — None means use flat Income Tax).
        # Activated for any country/region whose config sets pit_brackets.
        if configuration.pit_brackets is not None:
            brackets = np.array(configuration.pit_brackets, dtype=float)
            states["pit_thresholds"] = brackets[:, 0]
            states["pit_rates"] = brackets[:, 1]
            if configuration.pit_taxable_income_deductions is not None:
                states["pit_taxable_income_deductions"] = configuration.pit_taxable_income_deductions
            if configuration.pit_tax_credits is not None:
                states["pit_tax_credits"] = [
                    {
                        "kind": t.kind,
                        "amount": t.amount,
                        "indexing": t.indexing,
                        "age_min": t.eligibility_age_min,
                        "clawback_start": t.clawback_start,
                        "clawback_cap": t.clawback_cap,
                    }
                    for t in configuration.pit_tax_credits
                ]

        # Couple rental income split for progressive PIT
        states["couple_rental_income_split"] = configuration.couple_rental_income_split

        data = (synthetic_central_government.central_gov_data.astype(float)).rename_axis("Central Government ID")

        ts = create_central_government_timeseries(
            data=data,
            number_of_unemployed_individuals=number_of_unemployed_individuals,
        )

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    def reset(self, configuration: CentralGovernmentConfiguration):
        """Reset the government agent to initial state.

        Resets all state variables and updates function implementations
        based on the provided configuration.

        Args:
            configuration (CentralGovernmentConfiguration): New configuration
                parameters for the reset state
        """
        self.gen_reset()
        update_functions(
            model=configuration.functions, loc="macromodel.agents.central_government", functions=self.functions
        )

    def update_benefits(
        self,
        historic_ppi_inflation: list[np.ndarray],
        exogenous_ppi_inflation: np.ndarray,
        current_estimated_ppi_inflation: float,
        current_unemployment_rate: float,
        current_estimated_growth: float,
    ) -> None:
        """Update social benefit levels based on economic conditions.

        Adjusts both unemployment benefits and other social transfers
        considering:
        - Historical and expected inflation
        - Current unemployment rate
        - Economic growth estimates

        Args:
            historic_ppi_inflation (list[np.ndarray]): Past inflation rates
            exogenous_ppi_inflation (np.ndarray): External inflation factors
            current_estimated_ppi_inflation (float): Current inflation estimate
            current_unemployment_rate (float): Current unemployment rate
            current_estimated_growth (float): Estimated economic growth
        """
        all_ppi_inflation = np.concatenate(
            (
                exogenous_ppi_inflation,
                np.array(historic_ppi_inflation).flatten(),
                [current_estimated_ppi_inflation],
            )
        )

        # Unemployment benefits
        self.ts.unemployment_benefits_by_individual.append(
            [
                self.functions["social_benefits"].compute_unemployment_benefits(
                    prev_unemployment_benefits=self.ts.current("unemployment_benefits_by_individual")[0],
                    historic_ppi_inflation=all_ppi_inflation,
                    current_estimated_growth=current_estimated_growth,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["unemployment_benefits_model"],
                )
            ]
        )

        # Regular social transfers to households
        self.ts.total_other_benefits.append(
            [
                self.functions["social_benefits"].compute_regular_transfer_to_households(
                    prev_regular_transfer_to_households=self.ts.current("total_other_benefits")[0],
                    historic_ppi_inflation=all_ppi_inflation,
                    current_estimated_growth=current_estimated_growth,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["other_benefits_model"],
                )
            ]
        )

    def distribute_unemployment_benefits_to_individuals(
        self,
        current_individual_activity_status: np.ndarray,
    ) -> np.ndarray:
        """Distribute unemployment benefits to eligible individuals.

        Allocates unemployment benefits to individuals based on their
        current activity status (employed vs. unemployed).

        Args:
            current_individual_activity_status (np.ndarray): Activity status
                for each individual

        Returns:
            np.ndarray: Unemployment benefits by individual (zero for employed)
        """
        unemployment_benefits = np.zeros(current_individual_activity_status.shape)
        unemployment_benefits[current_individual_activity_status == ActivityStatus.UNEMPLOYED] = self.ts.current(
            "unemployment_benefits_by_individual"
        )[0]
        return unemployment_benefits.astype(float)

    def compute_taxes(
        self,
        current_ind_employee_income: np.ndarray,
        current_total_rent_paid: float,
        current_income_financial_assets: np.ndarray,
        current_ind_rental_income: np.ndarray | None = None,
        current_ind_financial_income: np.ndarray | None = None,
        current_ind_activity: np.ndarray = None,
        current_ind_realised_cons: np.ndarray = None,
        current_bank_profits: np.ndarray = None,
        current_firm_production: np.ndarray = None,
        current_firm_price: np.ndarray = None,
        current_firm_profits: np.ndarray = None,
        current_firm_industries: np.ndarray = None,
        current_household_new_real_wealth: np.ndarray = None,
        taxes_less_subsidies_rates: np.ndarray = None,
        current_total_exports: float = 0.0,
        individuals_age: np.ndarray | None = None,
        individuals_corr_households: np.ndarray | None = None,
        households_type: np.ndarray | None = None,
        households_n_adults: np.ndarray | None = None,
        children_under_18_per_ind: np.ndarray | None = None,
        children_under_6_per_ind: np.ndarray | None = None,
    ) -> None:
        """Calculate all tax revenues for the current period.

        Computes revenues from multiple tax sources:
        - Production and VAT
        - Income and corporate taxes
        - Social insurance contributions
        - Capital formation and export taxes

        When ``current_ind_rental_income`` and ``current_ind_financial_income``
        are provided (not None), the progressive PIT is applied to the
        combined individual taxable base:
        employee_income + rental_income + financial_income.

        Args:
            current_ind_employee_income (np.ndarray): Employee incomes per individual
            current_total_rent_paid (float): Total rent paid by renters (scalar)
            current_income_financial_assets (np.ndarray): Financial income per household
            current_ind_rental_income (Optional[np.ndarray]): Gross rental income per individual
            current_ind_financial_income (Optional[np.ndarray]): Financial income per individual
            current_ind_activity (np.ndarray): Individual activity status
            current_ind_realised_cons (np.ndarray): Consumption levels
            current_bank_profits (np.ndarray): Bank profits
            current_firm_production (np.ndarray): Firm production
            current_firm_price (np.ndarray): Product prices
            current_firm_profits (np.ndarray): Firm profits
            current_firm_industries (np.ndarray): Industry classifications
            current_household_new_real_wealth (np.ndarray): New wealth
            taxes_less_subsidies_rates (np.ndarray): Net tax rates
            current_total_exports (float): Total exports
            individuals_age (Optional[np.ndarray]): Age per individual
            individuals_corr_households (Optional[np.ndarray]): Household ID per individual
            households_type (Optional[np.ndarray]): HouseholdType enum per household
            households_n_adults (Optional[np.ndarray]): Number of adults per household
            children_under_18_per_ind (Optional[np.ndarray]): Children under 18 in individual's household
            children_under_6_per_ind (Optional[np.ndarray]): Children under 6 in individual's household
        """
        # Taxes on production
        self.ts.taxes_production.append(
            [np.sum(taxes_less_subsidies_rates[current_firm_industries] * current_firm_production * current_firm_price)]
        )

        # Value-added taxes
        self.ts.taxes_vat.append([self.states["Value-added Tax"] * np.sum(current_ind_realised_cons)])

        # Taxes on capital formation
        self.ts.taxes_cf.append(
            [self.states["Capital Formation Tax"] * np.sum(np.maximum(0.0, current_household_new_real_wealth))]
        )

        # Corporate income taxes
        self.ts.taxes_corporate_income.append(
            [
                self.states["Profit Tax"]
                * (np.sum(np.maximum(current_firm_profits, 0)) + np.sum(np.maximum(current_bank_profits, 0)))
            ]
        )

        # Taxes on exports
        self.ts.taxes_exports.append([self.states["Export Tax"] * current_total_exports])

        # Total wages of employed individuals (after Employee SI deduction —
        # this is the standard taxable base for personal income tax)
        tot_wages_employed_ind = np.sum([current_ind_employee_income[current_ind_activity == ActivityStatus.EMPLOYED]])

        # Build the individual-level taxable base for progressive PIT.
        taxable_base_per_ind = current_ind_employee_income * (
            1 - self.states["Employee Social Insurance Tax"]
        )

        # When rental and financial income are available at the individual
        # level (distributed from households), add them to the taxable base.
        use_individual_income = (
            current_ind_rental_income is not None
            and current_ind_financial_income is not None
        )
        if use_individual_income:
            taxable_base_per_ind = (
                taxable_base_per_ind
                + current_ind_rental_income
                + current_ind_financial_income
            )

        # Personal income tax: progressive when a schedule is
        # configured, otherwise flat on all income components.
        pit_thresholds = self.states.get("pit_thresholds")
        pit_rates = self.states.get("pit_rates")

        if pit_thresholds is not None and pit_rates is not None:
            # --- Progressive PIT on combined individual taxable base ---

            # Apply per-individual taxable-income deductions (reduces the
            # base before bracket thresholds, so it can drop a filer into
            # a lower bracket).
            pit_taxable_income_deductions = self.states.get("pit_taxable_income_deductions")
            taxable_income_for_brackets = taxable_base_per_ind
            if pit_taxable_income_deductions is not None and pit_taxable_income_deductions > 0:
                taxable_income_for_brackets = np.maximum(
                    0.0, taxable_base_per_ind - pit_taxable_income_deductions
                )

            pit_per_individual = compute_progressive_tax(
                taxable_income_for_brackets, pit_thresholds, pit_rates
            )

            # Apply non-refundable tax credits when configured.
            # Each component contributes its base amount for eligible
            # individuals (e.g. Personal Amount for all, Age Amount for
            # age ≥ 65).  Total credit = Σ eligible bases × pit_rates[0].
            # When no credits are configured, this is a no-op.
            pit_tax_credits = self.states.get("pit_tax_credits")

            if pit_tax_credits is not None and len(pit_tax_credits) > 0:
                n_ind = len(pit_per_individual)

                # --- Build per-individual household-context arrays ---
                # in_couple_household[i] = True if individual i lives in a
                # couple-type household (2 adults <65, 1 adult ≥65, couple
                # with/without children).
                in_couple_household = None
                is_single_parent = None
                spouse_income_per_ind = None  # spouse's taxable_base

                if (
                    individuals_corr_households is not None
                    and households_type is not None
                    and households_n_adults is not None
                ):
                    from macromodel.agents.households.household_properties import HouseholdType

                    # Couple households: TWO_ADULTS_YOUNGER_THAN_65, TWO_ADULTS_ONE_AT_LEAST_65,
                    # TWO_ADULTS_WITH_ONE_CHILD, TWO_ADULTS_WITH_TWO_CHILDREN,
                    # TWO_ADULTS_WITH_AT_LEAST_THREE_CHILDREN
                    couple_types = {
                        HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,
                        HouseholdType.TWO_ADULTS_ONE_AT_LEAST_65,
                        HouseholdType.TWO_ADULTS_WITH_ONE_CHILD,
                        HouseholdType.TWO_ADULTS_WITH_TWO_CHILDREN,
                        HouseholdType.TWO_ADULTS_WITH_AT_LEAST_THREE_CHILDREN,
                    }
                    single_parent_types = {
                        HouseholdType.SINGLE_PARENT_WITH_CHILDREN,
                    }

                    hh_id_of_ind = individuals_corr_households.astype(int)
                    hh_type_of_ind = np.array([
                        households_type[h] if h < len(households_type) else None
                        for h in hh_id_of_ind
                    ])

                    in_couple_household = np.array([
                        h_type in couple_types for h_type in hh_type_of_ind
                    ])
                    is_single_parent = np.array([
                        h_type in single_parent_types for h_type in hh_type_of_ind
                    ])

                    # Build spouse-income array for spousal amount clawback.
                    # Spouse income = taxable_base_per_ind of the other adult
                    # in the couple household (the one NOT claiming the credit).
                    # For individuals NOT in a couple household, spouse_income
                    # is set to a large number so the clawback never applies
                    # (0 credit after clawback, but they're already filtered by
                    # eligibility).
                    spouse_income_per_ind = np.full(n_ind, np.inf)
                    for hh_id in np.unique(hh_id_of_ind):
                        hh_mask = hh_id_of_ind == hh_id
                        hh_inds = np.where(hh_mask)[0]
                        if len(hh_inds) == 2 and hh_id < len(households_type):
                            hh_type = households_type[hh_id]
                            if hh_type in couple_types:
                                # Each spouse's "spouse_income" = the OTHER's taxable base
                                spouse_income_per_ind[hh_inds[0]] = taxable_base_per_ind[hh_inds[1]]
                                spouse_income_per_ind[hh_inds[1]] = taxable_base_per_ind[hh_inds[0]]

                per_ind_credit_base = np.zeros(n_ind)

                for tc in pit_tax_credits:
                    kind = tc["kind"]
                    amount = tc["amount"]
                    age_min = tc.get("age_min")

                    # ── eligibility mask ──
                    if age_min is not None and individuals_age is not None:
                        eligible = individuals_age >= age_min
                    elif kind == "Spousal Amount" and in_couple_household is not None:
                        eligible = in_couple_household.copy()
                    elif kind == "Equivalent To Spouse Amount" and is_single_parent is not None:
                        eligible = is_single_parent.copy()
                    elif kind.startswith("Child Amount"):
                        # Child credits are claimed by adults with children;
                        # amount is per-child.  Eligibility is gated by
                        # having ≥1 child of the relevant age (a parent).
                        eligible = np.ones(n_ind, dtype=bool)
                    else:
                        # Universal credit (e.g. Personal Amount)
                        eligible = np.ones(n_ind, dtype=bool)

                    # ── credit amount per individual ──
                    credit_amt = np.where(eligible, amount, 0.0)

                    # ── per-child multiplier ──
                    # Amount in the CSV is per eligible child.
                    if kind == "Child Amount Under 18" and children_under_18_per_ind is not None:
                        credit_amt = amount * children_under_18_per_ind
                    elif kind == "Child Amount Under 6" and children_under_6_per_ind is not None:
                        credit_amt = amount * children_under_6_per_ind

                    # ── Age Amount phaseout ──
                    # Formula: credit = max(0, amount − rate × max(0, own_income − clawback_start))
                    # where rate = amount / (cap − clawback_start), derived from the CSV data.
                    # Phaseout is against the individual's OWN taxable income.
                    if kind == "Age Amount":
                        cs = tc.get("clawback_start")
                        cc = tc.get("clawback_cap")
                        if cs is not None and cc is not None and cc > cs:
                            clawback_rate = amount / (cc - cs)
                            excess = np.maximum(0.0, taxable_base_per_ind - cs)
                            credit_amt = np.maximum(0.0, amount - clawback_rate * excess)

                    # ── spousal / eligible-dependant income test ──
                    # Real BC formula: credit = max(0, base_amount − other_income)
                    # where other_income = spouse's taxable income (spousal amount)
                    # or dependent's taxable income (equivalent to spouse amount).
                    if kind == "Spousal Amount" and spouse_income_per_ind is not None:
                        credit_amt = np.maximum(0.0, amount - spouse_income_per_ind)
                    elif kind == "Equivalent To Spouse Amount":
                        # Dependent has no income in the model → credit = full base
                        credit_amt = np.where(eligible, amount, 0.0)

                    per_ind_credit_base += credit_amt

                credit_per_ind = per_ind_credit_base * float(pit_rates[0])
                pit_per_individual = np.maximum(0.0, pit_per_individual - credit_per_ind)

            total_income_tax = pit_per_individual.sum()
            rental_tax_revenue = (
                current_ind_rental_income.sum() if use_individual_income
                else self.states["Income Tax"] * current_total_rent_paid
            )

            # Update the scalar effective rate so that behavioural decisions
            # (wage-setting, after-tax income, rental income) stay aligned
            # with the progressive schedule.
            total_taxable_base = taxable_base_per_ind.sum()
            if total_taxable_base > 0:
                self.states["Income Tax"] = float(total_income_tax / total_taxable_base)
        else:
            # --- Flat tax (backward-compatible path) ---
            total_income_tax = (
                self.states["Income Tax"] * (1 - self.states["Employee Social Insurance Tax"]) * tot_wages_employed_ind
                + self.states["Income Tax"] * current_total_rent_paid
                + self.states["Income Tax"] * current_income_financial_assets.sum()
            )
            rental_tax_revenue = self.states["Income Tax"] * current_total_rent_paid

        self.ts.taxes_income.append([total_income_tax])
        self.ts.taxes_rental_income.append([rental_tax_revenue])

        # Taxes on employer social insurance
        self.ts.taxes_employer_si.append([self.states["Employer Social Insurance Tax"] * tot_wages_employed_ind])

        # Taxes on employee social insurance
        self.ts.taxes_employee_si.append([self.states["Employee Social Insurance Tax"] * tot_wages_employed_ind])

    def compute_taxes_on_products(self) -> float:
        """Calculate total taxes on products and production.

        Aggregates various product-related taxes:
        - Production taxes
        - Value-added tax (VAT)
        - Capital formation tax
        - Export taxes

        Returns:
            float: Total tax revenue from products and production
        """
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_exports")[0]
        )

    def step_pit_brackets(
        self,
        tax_year: int,
        cpi_map: dict[int, float],
        base_year: int,
    ) -> None:
        """Inflate PIT thresholds, tax credits, and taxable-income
        deductions with compound CPI.

        Recomputes ``states["pit_thresholds"]``,
        ``states["pit_tax_credits"]``, and
        ``states["pit_taxable_income_deductions"]`` by compounding annual
        CPI inflation rates.  The nominal values stored at construction
        are never modified — inflation is always computed from those
        original values, making repeated calls safe.

        Call this once per simulated year (every 4 quarterly timesteps)
        to mirror real-world bracket indexation.

        Args:
            tax_year: Current tax year.
            cpi_map: ``{year: inflation_rate}`` mapping (0.018 = 1.8 %).
            base_year: Year whose thresholds are the nominal base.
        """
        if self.pit_base_thresholds is None:
            return
        if tax_year <= base_year or not cpi_map:
            return

        factor = 1.0
        for y in range(base_year, tax_year):
            rate = cpi_map.get(y)
            if rate is not None:
                factor *= 1.0 + rate

        self.states["pit_thresholds"] = self.pit_base_thresholds * factor

        if self.pit_base_deductions is not None:
            self.states["pit_taxable_income_deductions"] = self.pit_base_deductions * factor

        # CPI-inflate tax credit components (only those with indexing=True).
        if self.pit_base_tax_credits is not None:
            inflated = []
            for tc in self.pit_base_tax_credits:
                entry = {
                    "kind": tc["kind"],
                    "amount": tc["amount"] * factor if tc.get("indexing", True) else tc["amount"],
                    "indexing": tc.get("indexing", True),
                    "age_min": tc.get("age_min"),
                }
                # Only include clawback keys if they were in the base dict
                if "clawback_start" in tc:
                    tc_cs = tc["clawback_start"]
                    entry["clawback_start"] = tc_cs * factor if (tc.get("indexing", True) and tc_cs is not None) else tc_cs
                if "clawback_cap" in tc:
                    tc_cc = tc["clawback_cap"]
                    entry["clawback_cap"] = tc_cc * factor if (tc.get("indexing", True) and tc_cc is not None) else tc_cc
                inflated.append(entry)
            self.states["pit_tax_credits"] = inflated

    def compute_revenue(
        self,
        household_rent_paid_to_government: float,
    ) -> float:
        """Calculate total government revenue.

        Aggregates all revenue sources:
        - All tax revenues
        - Social insurance contributions
        - Rental income from public housing

        Args:
            household_rent_paid_to_government (float): Rent from public housing

        Returns:
            float: Total government revenue
        """
        self.ts.total_rent_received.append([household_rent_paid_to_government])
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_corporate_income")[0]
            + self.ts.current("taxes_exports")[0]
            + self.ts.current("taxes_income")[0]
            + self.ts.current("taxes_employee_si")[0]
            + self.ts.current("taxes_employer_si")[0]
            + household_rent_paid_to_government
        )

    def compute_deficit(
        self,
        current_ind_activity: np.ndarray,
        current_household_social_transfers: np.ndarray,
        current_government_nominal_amount_spent: np.ndarray,
        government_interest_rates: float,
    ) -> np.ndarray:
        """Calculate the government deficit.

        Computes deficit as the difference between:
        Expenditures:
        - Unemployment benefits
        - Social transfers
        - Government spending
        - Interest payments
        And:
        - Total revenue

        Args:
            current_ind_activity (np.ndarray): Individual activity status
            current_household_social_transfers (np.ndarray): Social transfers
            current_government_nominal_amount_spent (np.ndarray): Spending
            government_interest_rates (float): Interest rate on debt

        Returns:
            np.ndarray: Government deficit (positive = deficit)
        """
        total_unemployment_benefits = (
            np.sum(current_ind_activity == ActivityStatus.UNEMPLOYED)
            * self.ts.current("unemployment_benefits_by_individual")[0]
        )
        total_household_social_transfers = np.sum(current_household_social_transfers)
        all_benefits = total_unemployment_benefits + total_household_social_transfers
        interest_payments = government_interest_rates * self.ts.current("debt")[0]
        return np.array(
            [
                all_benefits
                + np.sum(current_government_nominal_amount_spent)
                + interest_payments
                - self.ts.current("revenue")[0]
            ]
        )

    def compute_debt(self) -> np.ndarray:
        """Update government debt level.

        Calculates new debt level by adding current deficit
        to existing debt stock.

        Returns:
            np.ndarray: Updated government debt level
        """
        return np.array([self.ts.current("debt")[0] + self.ts.current("deficit")[0]])

    def save_to_h5(self, group: h5py.Group):
        """Save government data to HDF5 format.

        Stores all time series data in the specified HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to save data in
        """
        self.ts.write_to_h5("central_government", group)

    def total_taxes(self):
        """Calculate total tax revenue on products.

        Returns:
            float: Aggregate tax revenue from all product-related taxes
        """
        return self.ts.get_aggregate("taxes_on_products")
