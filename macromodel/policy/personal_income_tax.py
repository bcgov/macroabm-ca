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

    def __init__(self, brackets: list[tuple[float, float]], scale: int, increments_per_year: int):
        """Initialize abstract personal income tax policy.
        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
            scale (int): Scaling factor for population-based calculations
            increments_per_year (int): Timesteps per year

        """
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year

    @abstractmethod
    def compute_individual_tax(self, taxable_income: float) -> float:
        """Calculate individual tax based on income."""
        pass

    @abstractmethod
    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Set the taxation rate for this timestep"""
        pass

    def get_rate(self) -> list[tuple[float, float]]:
        """Return the taxation rate for this timestep
        
        Returns:
            list[tuple[float, float]]: Taxation rate as recorded as (threshold, rate)
        """
        return self.brackets

    def compute_withhold(
            self,
            income_component_by_individual: np.ndarray, 
            ) -> np.ndarray:
        """Calculate estimate of income to be withheld to pay personal income tax for one timestep
        
        Attributes:
            income_component_by_individual (np.ndarray): Income component (ie real wages or unemployment benefits)

        Returns:
            np.ndarray: Estimate of personal income tax owed
        """

        # blanks for timeseries
        individuals_personal_income_tax_owed = np.zeros_like(income_component_by_individual)

        # estimated component of personal income tax owed
        for i in range(len(income_component_by_individual)):
            projected_annual_tax= self.compute_individual_tax(
                income_component_by_individual[i]/self.scale * self.increments_per_year
                ) * self.scale
            individuals_personal_income_tax_owed[i] = projected_annual_tax / self.increments_per_year

        return individuals_personal_income_tax_owed 
    
    def compute_personal_income_tax_owed(
            self, 
            individuals: Individuals, 
            households: Households, 
            ) -> float:
        """Calculate personal income tax owed for all indiviudal agents for one timestep
        
        Attributes:
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used

        Returns:
           float: Total personal income tax owed by all individual agents for this timestep
        """                   

        debug_individuals = False
        debug_total_components = False

        # blanks for timeseries
        individuals_taxable_income = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_employment = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_unemployment = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_rental = np.zeros_like(individuals.ts.current("employee_income"))
        # individuals_taxable_income_investment = np.zeros_like(individuals.ts.current("employee_income"))  # TODO: implement
        individuals_personal_income_tax_owed = np.zeros_like(individuals.ts.current("employee_income"))

        # calculate all taxable income streams across individuals
        for i in range(individuals.n_individuals):
            if individuals.ts.current("employee_income_gross")[i] < 0:
                raise ValueError("employee_income cannot be negative.")
            # TODO: future extension will mean individuals_taxable_income_employment != employee_income_gross
            #       this is come when a feature to convert total income -> net income -> taxable income
            individuals_taxable_income_employment[i] = individuals.ts.current("employee_income_gross")[i]   

            # NOTE: evidence suggests "income_from_unemployment_benefits" isn't functioning properly
            # - almost always zero even when unemployed
            # TODO: check against central_government.ts.current("unemployment_benefits_by_individual")[0] in compute_deficit()
            if individuals.ts.current("income_from_unemployment_benefits")[i] < 0:
                raise ValueError("income_from_unemployment_benefits cannot be negative.")
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
                if households.ts.current("income_rental")[hh_id] < 0:
                    raise ValueError("income_rental cannot be negative.")
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
                individuals_taxable_income[i]/self.scale * self.increments_per_year
                ) * self.scale
            individuals_personal_income_tax_owed[i] = projected_annual_tax / self.increments_per_year

            if debug_individuals:    # !UNSCALED! diagnostic so that it can be compared to real world individuals
                print(
                    f"{i}: " +
                    f"hh_id: {individuals.states["Corresponding Household ID"][i]}, " +
                    f"employed: {individuals.states["Activity Status"][i] == ActivityStatus.EMPLOYED}, " +
                    f"income_employment_unscaled_annual: ${individuals_taxable_income_employment[i]/self.scale * self.increments_per_year:,.2f}, " + 
                    f"income_unemployment_unscaled_annual: ${individuals_taxable_income_unemployment[i]/self.scale * self.increments_per_year:,.2f}, " + 
                    f"income_rental_unscaled_annual: ${individuals_taxable_income_rental[i]/self.scale * self.increments_per_year:,.2f}, " +
                    f"taxable_income_unscaled_annual: ${individuals_taxable_income[i]/self.scale * self.increments_per_year:,.2f}, " + 
                    f"tax owed_unscaled_annual: ${individuals_personal_income_tax_owed[i]/self.scale * self.increments_per_year:,.2f}"
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

    def compute_annual_personal_income_tax_owed(
            self, 
            timestep: Timestep, 
            individuals: Individuals, 
            households: Households, 
            ) -> float:
        """Calculate total annual personal income tax of all indiviudal agents 
            and reconcile with those collected within this year
        
        Attributes:
            timestep (Timestep): model timestep
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used

        Returns:
           float: Total annual personal income tax owed by all individual agents for this year
        """                   

        debug_individuals = False
        debug_total_components = False

        if timestep.month != 10:
            raise ValueError("Annual tax is only calculated at year end.")

        total_annual_tax = 0

        # blanks for timeseries
        annual_taxable_income = np.zeros_like(individuals.ts.current("employee_income"))
        annual_personal_income_tax_paid = np.zeros_like(individuals.ts.current("employee_income"))
        annual_personal_income_tax_owed = np.zeros_like(individuals.ts.current("employee_income"))
        personal_income_tax_adjustment = np.zeros_like(individuals.ts.current("employee_income"))

        t = len(individuals.ts.employee_income) - 1     # current timestep index
        for i in range(individuals.n_individuals):
            for s in range(self.increments_per_year):
                # calculte taxes already paid for the year
                annual_personal_income_tax_paid[i] += individuals.ts.personal_income_tax_paid[t - s][i]

                # calculate actual annual earnings
                annual_taxable_income[i] += individuals.ts.taxable_income[t - s][i]              

            # calculate individual tax owed
            # TODO: formulate a good way to store this that aligns with existing quarterly timeseries structure (if useful?)
            annual_personal_income_tax_owed[i] = self.compute_individual_tax(annual_taxable_income[i] / self.scale) * self.scale
            total_annual_tax += annual_personal_income_tax_owed[i]
                        
            # calculat the difference between amount paid vs due
            personal_income_tax_adjustment[i] = annual_personal_income_tax_owed[i] - annual_personal_income_tax_paid[i]

        # reconcile difference
        individuals.ts.personal_income_tax_adjustment.append(personal_income_tax_adjustment)
    
        # !!!CAREFUL!!! overwrite timeseries entries after iterating over individuals
        original_employee_income = np.copy(individuals.ts.current("employee_income"))
        original_personal_income_tax_paid = np.copy(individuals.ts.current("personal_income_tax_paid"))
        original_income_from_unemployment_benefits = np.copy(individuals.ts.current("income_from_unemployment_benefits"))
        for i in range(individuals.n_individuals):
            # NOTE: even if the individual changes "Activity Status" between timesteps the only state that matters
            #       for reconciliation is the current one (ie where to put the money so that it has impact on current spending)
            if individuals.states["Activity Status"][i] == ActivityStatus.EMPLOYED:
                individuals.ts.dicts["employee_income"][-1][i] = (
                    original_employee_income[i] - personal_income_tax_adjustment[i]
                )
            elif individuals.states["Activity Status"][i] == ActivityStatus.UNEMPLOYED:
                individuals.ts.dicts["income_from_unemployment_benefits"][-1][i] = (
                    original_income_from_unemployment_benefits[i] - personal_income_tax_adjustment[i]
                )
            
            individuals.ts.dicts["personal_income_tax_paid"][-1][i] = (
                original_personal_income_tax_paid[i] + personal_income_tax_adjustment[i]
            )

        if debug_total_components:
                    print(f"total_annual_tax: ${total_annual_tax:,.2f}")

        # TODO: compare total_annual_tax to central_government.ts.current("taxes_income")
        return total_annual_tax

class FlatRate(PersonalIncomeTax):
    """Flat rate income tax policy."""
    def __init__(self, brackets: list[tuple[float, float]], scale: int, increments_per_year: int):
        """Initialize Flat rate income tax policy.
        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
                Example [(float('inf'), 0.15)]
                Only threshold must be float('inf')
            scale (int): Scaling factor for population-based calculations
            increments_per_year (int): Timesteps per year

        """
        self._validate_brackets(brackets)
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year

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

    # def get_rate(self) -> list[tuple[float, float]]:
    #     """Return the taxation rate for this timestep

    #     Returns:
    #         list[tuple[float, float]]: Taxation rate as recorded as (threshold, rate)
    #     """
    #     return self.brackets

    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Return the taxation rate for this timestep
        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
        """
        self._validate_brackets(brackets)        
        self.brackets = brackets

class ProgressiveRate(PersonalIncomeTax):
    """Progressive rate income tax policy with brackets."""
    def __init__(self, brackets: list[tuple[float, float]], scale: int, increments_per_year: int):
        """Initialize Progressive rate income tax policy.
                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
                Example [(50000, 0.1), (100000, 0.2), (float('inf'), 0.3)] 
                Final threshold must be float('inf')
            scale (int): Scaling factor for population-based calculations
            increments_per_year (int): Timesteps per year 
        """

        self._validate_brackets(brackets)
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year

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

    # def get_rate(self) -> list[tuple[float, float]]:
    #     """Return the taxation rate for this timestep

    #     Returns:
    #         list[tuple[float, float]]: Taxation rate as recorded as (threshold, rate)
    #     """
    #     return self.brackets
    
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
