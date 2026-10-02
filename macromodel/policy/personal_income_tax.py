from abc import ABC, abstractmethod
from typing import Protocol
from macromodel.timestep import Timestep
from macromodel.agents.households.households import Households
from macromodel.agents.individuals.individuals import Individuals
from macromodel.agents.individuals.individual_properties import ActivityStatus
import numpy as np
# import inspect

class PersonalIncomeTax(ABC):
    """Abstract base class for personal income tax policy implementations.
    
    Defines interface for determining personal income tax through:
        - Computing tax owed by individual agents
        - Reviewing taxation rates
        - Updating taxation rates
        - Computing total tax imputed by all individual agents
    """

    @abstractmethod
    def compute_individual_tax(self, taxable_income: float) -> float:
        """Calculate individual tax based on income."""
        pass

    @abstractmethod
    def get_rate(self) -> list[tuple[float, float]]:
        """Return the taxation rate for this timestep"""
        pass

    @abstractmethod
    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Set the taxation rate for this timestep"""
        pass

    def compute_timestep_tax(
            self, 
            timestep: Timestep, 
            individuals: Individuals, 
            households: Households, 
            scale: int
            ) -> float:
        """Calculate personal income tax of all indiviudal agents for one timestep
        
        Attributes:
            timestep (Timestep): model timestep
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used
            scale (int): Number of people per individual agent

        Returns:
           float: Total personal income tax owed by all individual agents for this timestep
        """                   

        debug_individuals = False
        debug_total_components = False
        increments_per_year = int(12 / timestep.increment)

        # blanks for timeseries
        individuals_taxable_income = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_employment = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_unemployment = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_rental = np.zeros_like(individuals.ts.current("employee_income"))
        # individuals_taxable_income_investment = np.zeros_like(individuals.ts.current("employee_income"))  # TODO: implement
        individuals_personal_income_tax_owed = np.zeros_like(individuals.ts.current("employee_income"))

        # calculate all taxable income streams across individuals
        for i in range(individuals.n_individuals):
            individuals_taxable_income_employment[i] = individuals.ts.current("employee_income")[i]

            # NOTE: evidence suggests "income_from_unemployment_benefits" isn't functioning properly
            # - almost always zero even when unemployed
            # TODO: check against central_government.ts.current("unemployment_benefits_by_individual")[0] in compute_deficit()
            individuals_taxable_income_unemployment[i] = individuals.ts.current("income_from_unemployment_benefits")[i]

            # caclulate rental income (assuming it is split between all adult residents)
            # NOTE: alternative method could be to assume split between all employed adults
            # NOTE: evidence suggests "income_rental" isn't functioning properly 
            # - "income_rental" is always zero 
            #   WHILE self.households.ts.current("rent")[self.households.states["Tenure Status of the Main Residence"] == 3].sum()
            #   is > 0
            individuals_taxable_income_rental[i] = 0
            if individuals.states["Corresponding Household ID"][i] != 0:       # check if belongs to a hh
                hh_id = individuals.states["Corresponding Household ID"][i]    # identify which hh it belongs to

                # assume income_rental is split evenly between all adults
                if households.ts.current("income_rental")[hh_id] > 0 and individuals.states["Age"][i] >= 18:
                    individuals_taxable_income_rental[i] += (
                        households.ts.current("income_rental")[hh_id] / 
                        households.states["Number of Adults"][hh_id]
                    )

            individuals_taxable_income[i] = (
                individuals_taxable_income_employment[i] +
                individuals_taxable_income_unemployment[i] +
                individuals_taxable_income_rental[i]
            )

            # calculate individual tax owed 
            # (assuming individual earns current timestep individuals_taxable_income for all increments of year)
            # NOTE: individuals_taxable_income must be !UNSCALED! to align with values in self.brackets
            projected_annual_tax = self.compute_individual_tax(
                individuals_taxable_income[i]/scale * increments_per_year
                ) * scale
            individuals_personal_income_tax_owed[i] = projected_annual_tax / increments_per_year

            if debug_individuals:    # !UNSCALED! diagnostic so that it can be compared to real world individuals
                print(
                    f"{i}: " +
                    f"hh_id: {individuals.states["Corresponding Household ID"][i]}, " +
                    f"employed: {individuals.states["Activity Status"][i] == ActivityStatus.EMPLOYED}, " +
                    f"income_employment: ${individuals_taxable_income_employment[i]/scale * increments_per_year:,.2f}, " + 
                    f"income_unemployment: ${individuals_taxable_income_unemployment[i]/scale * increments_per_year:,.2f}, " + 
                    f"income_rental: ${individuals_taxable_income_rental[i]/scale * increments_per_year:,.2f}, " +
                    f"taxable_income: ${individuals_taxable_income[i]/scale * increments_per_year:,.2f}, " + 
                    f"tax owed: ${individuals_personal_income_tax_owed[i]/scale * increments_per_year:,.2f}"
                    )
            
        if debug_total_components:  # !SCALED! diagnostic so that it can be compared to provincial values
            print(
                f"total income_employment: ${np.sum(individuals_taxable_income_employment):,.2f}, " + 
                f"total income_unemployment: ${np.sum(individuals_taxable_income_unemployment):,.2f}, " + 
                f"total income_rental: ${np.sum(individuals_taxable_income_rental):,.2f}, " +
                f"total taxable_income: ${np.sum(individuals_taxable_income):,.2f}" 
                )

        # store income streams and tax owed by all individuals
        individuals.ts.taxable_income.append(individuals_taxable_income)
        individuals.ts.taxable_income_employment.append(individuals_taxable_income_employment)
        individuals.ts.taxable_income_unemployment.append(individuals_taxable_income_unemployment)
        individuals.ts.taxable_income_rental.append(individuals_taxable_income_rental)
        individuals.ts.personal_income_tax_owed.append(individuals_personal_income_tax_owed)

        if debug_total_components:
            print(f"Total personal_income_tax_owed: ${np.sum(individuals_personal_income_tax_owed):,.2f} " + 
                  f"collected from {individuals.n_individuals} individuals"
                  )

        return np.sum(individuals_personal_income_tax_owed)

    def compute_annual_tax(
            self, 
            timestep: Timestep, 
            individuals: Individuals, 
            households: Households, 
            scale: int
            ) -> float:
        """Calculate total annual personal income tax of all indiviudal agents 
            and reconcile with those collected within this year
        
        Attributes:
            timestep (Timestep): model timestep
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used
            scale (int): Number of people per individual agent

        Returns:
           float: Total annual personal income tax owed by all individual agents for this year
        """                   

        debug_individuals = False
        debug_total_components = False
        increments_per_year = int(12 / timestep.increment)

        if timestep.month != 10:
            raise ValueError("Annual tax is only calculated at year end.")

        total_annual_tax = 0

        t = len(individuals.ts.employee_income) - 1     # current timestep index
        for i in range(individuals.n_individuals):
            annual_personal_income_tax_paid = 0
            annual_taxable_income = 0
            for s in range(increments_per_year):
                # calculte taxes already paid for the year
                annual_personal_income_tax_paid += individuals.ts.personal_income_tax_owed[t - s][i]

                # calculate taxes that are due based on actual annual earnings
                annual_taxable_income += individuals.ts.taxable_income[t - s][i]              

            # calculate individual tax owed
            # TODO: formulate a good way to store this that aligns with existing quarterly timeseries structure (if useful?)
            annual_personal_income_tax_due = self.compute_individual_tax(annual_taxable_income / scale) * scale
            total_annual_tax += annual_personal_income_tax_due
            
            # reconcile difference between amount collected vs due
            # TODO: store tax_adjustment if used elsewhere 
            #   - could be useful to adjust individual / hh wealth to regain stock-flow consistency (or for diagnostics?)
            tax_adjustment = annual_personal_income_tax_due - annual_personal_income_tax_paid
            original_payment = individuals.ts.current("personal_income_tax_owed")[i]

            # !!!CAREFUL!!! overwrite timeseries entry
            individuals.ts.dicts["personal_income_tax_owed"][-1][i] = original_payment + tax_adjustment

        if debug_total_components:
                    print(f"total_annual_tax: ${total_annual_tax:,.2f}")

        # TODO: compare total_annual_tax to central_government.ts.current("taxes_income")
        return total_annual_tax

class FlatRate(PersonalIncomeTax):
    """Flat rate income tax policy."""
    def __init__(self, brackets: list[tuple[float, float]]):
        """Initialize Flat rate income tax policy.
        
        Attributes:
            brackets (list[tuple[float, float]]): taxation rate as recorded as (threshold, rate)
                Example [(float('inf'), 0.15)]
                Only threshold must be float('inf') 
        """
        self._validate_brackets(brackets)
        self.brackets = brackets

    def _validate_brackets(self, brackets: list[tuple[float, float]]) -> None:
        """Ensure brackets are valid
                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
        """
        if not brackets or len(brackets) != 1:
            raise ValueError("FlatRate must have exactly one bracket.")
        
        threshold, rate = brackets[0]

        if threshold != float('inf'):
            raise ValueError("FlatRate threshold must be float('inf').")
        if not (0 <= rate <= 1):
            raise ValueError("rate must be between 0 and 1.")

    def compute_individual_tax(self, taxable_income: float) -> float:
        """Calculate the personal income tax owed by an individual agent
                        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)

        Returns:
            float: Personal income tax owed by an individual agent
        """
        if taxable_income < 0:
            raise ValueError("Taxable income cannot be negative.")
        _, rate = self.brackets[0]

        return taxable_income * rate

    def get_rate(self) -> list[tuple[float, float]]:
        """Return the taxation rate for this timestep

        Returns:
            list[tuple[float, float]]: Taxation rate as recorded as (threshold, rate)
        """
        return self.brackets

    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Return the taxation rate for this timestep
        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
        """
        self._validate_brackets(brackets)        
        self.brackets = brackets

class ProgressiveRate(PersonalIncomeTax):
    """Progressive rate income tax policy with brackets."""
    def __init__(self, brackets: list[tuple[float, float]]):
        """Initialize Progressive rate income tax policy.
                
        Attributes:
            brackets (list[tuple[float, float]]): taxation rate as recorded as (threshold, rate)
                Example [(50000, 0.1), (100000, 0.2), (float('inf'), 0.3)] 
                Final threshold must be float('inf') 
        """

        self._validate_brackets(brackets)
        self.brackets = brackets

    def _validate_brackets(self, brackets: list[tuple[float, float]]) -> None:
        """Ensure brackets are valid
                        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
        """
        if not brackets:
            raise ValueError("ProgressiveRate brackets cannot be empty.")
        if len(brackets) < 2:
            raise ValueError("ProgressiveRate must have more than one bracket.")
        if brackets[-1][0] != float('inf'):
            raise ValueError("ProgressiveRate final bracket threshold must be float('inf').")
        for threshold, rate in brackets:
            if threshold <= 0 and threshold != float('inf'):
                raise ValueError("ProgressiveRate thresholds must be positive or float('inf').")
            if not (0 <= rate <= 1):
                raise ValueError("Tax rate must be between 0 and 1.")
    
    def compute_individual_tax(self, taxable_income: float) -> float:
        """Calculate the personal income tax owed by an individual agent
                                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)

        Returns:
            float: Personal income tax owed by an individual agent
        """
        if taxable_income < 0:
            raise ValueError("Taxable income cannot be negative.")
        tax = 0.0
        prev_threshold = 0.0
        for threshold, rate in self.brackets:
            taxable = min(taxable_income, threshold) - prev_threshold
            if taxable > 0:
                tax += taxable * rate
            prev_threshold = threshold
            if taxable_income <= threshold:
                break
            
        return tax

    def get_rate(self) -> list[tuple[float, float]]:
        """Return the taxation rate for this timestep

        Returns:
            list[tuple[float, float]]: Taxation rate as recorded as (threshold, rate)
        """
        return self.brackets
    
    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Return the taxation rate for this timestep
                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
        """
        self._validate_brackets(brackets)
        self.brackets = brackets

class PersonalIncomeTaxProtocol(Protocol):
    def compute_individual_tax(self, taxable_income: float) -> float: ...
    def get_rate(self) -> list[tuple[float, float]]: ...
    def set_rate(self, brackets: list[tuple[float, float]]) -> None: ...
