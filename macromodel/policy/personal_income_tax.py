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
        """Calculate total personal income tax of all indiviudal agents for one timestep
        
        Attributes:
            timestep (Timestep): model timestep
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used
            scale (int): Number of people per individual agent

        Returns:
           float: Total timestep personal income tax owed by all individual agents
        """                   

        debug_individual = False
        debug_total_components = False
        increments_per_year = int(12 / timestep.increment)

        # personal income tax streams
        total_timestep_income = 0
        total_timestep_income_employment = 0
        total_timestep_income_unemployment = 0
        total_timestep_income_rental = 0

        individuals_timestep_tax = np.zeros_like(individuals.ts.current("employee_income"))

        # output
        total_timestep_tax = 0

        # sum all taxable income streams across individuals
        for i in range(individuals.n_individuals):
            timestep_income_employment = individuals.ts.current("employee_income")[i] / scale

            # NOTE: evidence suggests "income_from_unemployment_benefits" isn't functioning properly
            # - almost always zero even when unemployed
            timestep_income_unemployment = individuals.ts.current("income_from_unemployment_benefits")[i] / scale

            # caclulate rental income (assuming it is split between all adult residents)
            # NOTE: alternative method could be to assume split between all employed adults
            # NOTE: evidence suggests "income_rental" isn't functioning properly 
            # - "income_rental" is always zero 
            #   WHILE self.households.ts.current("rent")[self.households.states["Tenure Status of the Main Residence"] == 3].sum()
            #   is > 0
            timestep_income_rental = 0

            if individuals.states["Corresponding Household ID"][i] != 0:       # check if belongs to a hh
                hh_id = individuals.states["Corresponding Household ID"][i]    # identify which hh it belongs to
                
                if households.ts.current("income_rental")[hh_id] > 0 and individuals.states["Age"][i] >= 18:
                    timestep_income_rental += (
                        households.ts.current("income_rental")[hh_id] / 
                        households.states["Number of Adults"][hh_id] / 
                        scale
                    )
                    print("ALERT: Found a non-zero self.households.ts.income_rental")   # hh rental income reported!!!

            timestep_income = (
                timestep_income_employment +
                timestep_income_unemployment +
                timestep_income_rental
            )

            # calculate individual tax owed 
            # (assuming individual earns current timestep_individual_income for all increments of year)
            projected_annual_tax = self.compute_individual_tax(timestep_income * increments_per_year)
            projected_timestep_tax = projected_annual_tax / increments_per_year

            # overwrite tax owed by individual for this timestep
            individuals_timestep_tax[i] = projected_timestep_tax

            # sum components
            total_timestep_income += timestep_income
            total_timestep_income_employment += timestep_income_employment
            total_timestep_income_unemployment += timestep_income_unemployment
            total_timestep_income_rental += timestep_income_rental
            total_timestep_tax += projected_timestep_tax

            # diagnostics
            if debug_individual:
                print(
                    f"{i}: " +
                    f"hh_id: {individuals.states["Corresponding Household ID"][i]}, " +
                    f"employed: {individuals.states["Activity Status"][i] == ActivityStatus.EMPLOYED}, " +
                    f"proj inc_empl: ${timestep_income_employment * increments_per_year:,.2f}, " + 
                    f"proj inc_unemp: ${timestep_income_unemployment * increments_per_year:,.2f}, " + 
                    f"proj inc_rent: ${timestep_income_rental * increments_per_year:,.2f}, " +
                    f"proj inc_tot: ${timestep_income * increments_per_year:,.2f}, " + 
                    f"proj tax: ${projected_timestep_tax * increments_per_year:,.2f}"
                    )
            
        if debug_total_components:
            print(
                f"actual tot_inc_empl: ${total_timestep_income_employment:,.2f}, " + 
                f"actual tot_inc_unemp: ${total_timestep_income_unemployment:,.2f}, " + 
                f"actual tot_inc_rent: ${total_timestep_income_rental:,.2f}, " +
                f"actual tot_inc_tot: ${total_timestep_income:,.2f}" 
                )

        # store tax owed by all individuals
        individuals.ts.personal_income_tax_paid.append(individuals_timestep_tax)

        total_timestep_tax *= scale     # scale to model

        if debug_total_components:
            print(f"Total income tax: ${total_timestep_tax:,.2f} collected from {individuals.n_individuals} individuals")

        return total_timestep_tax

    def compute_annual_tax(
            self, 
            timestep: Timestep, 
            individuals: Individuals, 
            households: Households, 
            scale: int
            ) -> float:
        """Calculate total annual personal income tax of all indiviudal agents 
            and reconcile with timestep taxes collected within this year
        
        Attributes:
            timestep (Timestep): model timestep
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used
            scale (int): Number of people per individual agent

        Returns:
           float: Total annual personal income tax owed by all individual agents
        """                   

        debug_individual = False
        debug_total_components = False
        increments_per_year = int(12 / timestep.increment)

        if timestep.month != 10:
            raise ValueError("Annual tax can only be calculated at year end.")

        total_annual_tax = 0

        t = len(individuals.ts.employee_income) - 1
        for i in range(individuals.n_individuals):
            personal_income_tax_paid = 0
            annual_income_employment = 0
            annual_income_unemployment = 0
            annual_income_rental = 0

            # calculte taxes already paid for the year
            for s in range(increments_per_year):
                personal_income_tax_paid += individuals.ts.personal_income_tax_paid[t - s][i] / scale

            # calculate taxes that are due based on actual annual earnings
            for s in range(increments_per_year):
                annual_income_employment += individuals.ts.personal_income_tax_paid[t - s][i] / scale
                annual_income_unemployment += individuals.ts.income_from_unemployment_benefits[t - s][i] / scale
            
                if individuals.states["Corresponding Household ID"][i] != 0:       # check if belongs to a hh
                    hh_id = individuals.states["Corresponding Household ID"][i]    # identify which hh it belongs to
                    
                    if households.ts.current("income_rental")[hh_id] > 0 and individuals.states["Age"][i] >= 18:
                        annual_income_rental += (
                            households.ts.income_rental[t - s][hh_id] / 
                            households.states["Number of Adults"][hh_id] / 
                            scale
                        )
                        print("ALERT: Found a non-zero self.households.ts.income_rental")   # hh rental income reported!!!

            annual_income = (
                annual_income_employment +
                annual_income_unemployment +
                annual_income_rental
            )

            # calculate individual tax owed
            # TODO: formulate a good way to store this that aligns with existing quarterly timeseries structure (if useful?)
            annual_tax = self.compute_individual_tax(annual_income)
            total_annual_tax += annual_tax
            
            # reconcile difference between amount collected vs due
            # TODO: store value if used elsewhere (or for diagnostics?)
            # TODO: use to adjust individual / hh wealth to regain stock-flow consistency
            tax_adjustment = (annual_tax - personal_income_tax_paid)* scale
            original_payment = individuals.ts.current("personal_income_tax_paid")[i]

            # !!!CAREFUL!!! overwrite timeseries entry
            individuals.ts.dicts["personal_income_tax_paid"][-1][i] = original_payment + tax_adjustment

        total_annual_tax *= scale     # scale to model

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
