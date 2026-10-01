from abc import ABC, abstractmethod
from typing import Protocol
from macromodel.agents.households.households import Households
from macromodel.agents.individuals.individuals import Individuals
from macromodel.agents.individuals.individual_properties import ActivityStatus
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
        """Return the taxation rate."""
        pass

    @abstractmethod
    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Set the taxation rate."""
        pass

    def compute_total_quarterly_tax(self, individuals: Individuals, households: Households, scale: int) -> float:
        """Calculate total tax of all indiviudal agents
        
        Attributes:
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used
            scale (int): Number of people per individual agent
        
        """                   

        debug_individual = False
        debug_total_components = True

        ### assemble all personal income tax streams
        total_tax = 0
        total_taxable_income = 0
        total_income_employment = 0
        total_income_unemployment = 0
        total_income_rental = 0

        # sum all taxable income streams across individuals
        for i in range(individuals.n_individuals):
            income_employment = individuals.ts.current("employee_income")[i] / scale
            income_unemployment = individuals.ts.current("income_from_unemployment_benefits")[i] / scale

            # caclulate rental income
            income_rental = 0
            if individuals.states["Corresponding Household ID"][i] != 0:       # check if belongs to a hh
                hh_id = individuals.states["Corresponding Household ID"][i]    # identify which hh it belongs to
                if households.ts.current("income_rental")[hh_id] > 0:
                    income_rental += (
                        households.ts.current("income_rental")[hh_id] / 
                        households.states["Number of Adults"][hh_id] / 
                        scale
                    )
                    print("ALERT: Found a non-zero self.households.ts.income_rental")

            individual_taxable_income = (
                income_employment +
                income_unemployment +
                income_rental
            )

            # calculate individual tax owed
            individual_tax = self.compute_individual_tax(individual_taxable_income)

            # store components
            total_taxable_income += individual_taxable_income
            total_income_employment += income_employment
            total_income_unemployment += income_unemployment
            total_income_rental += income_rental
            total_tax += individual_tax

            # diagnostics
            if debug_individual:
                print(
                    f"{i}: " +
                    f"hh_id: {individuals.states["Corresponding Household ID"][i]}, " +
                    f"employed: {individuals.states["Activity Status"][i] == ActivityStatus.EMPLOYED}, " +
                    f"in_empl: ${income_employment:,.2f}, " + 
                    f"in_unemp: ${income_unemployment:,.2f}, " + 
                    f"in_rent: ${income_rental:,.2f}, " +
                    f"in_tot: ${individual_taxable_income:,.2f}, " + 
                    f"tax: ${individual_tax:,.2f}"
                    )
            
        if debug_total_components:
            print(
                f"tot_in_empl: ${total_income_employment:,.2f}, " + 
                f"tot_in_unemp: ${total_income_unemployment:,.2f}, " + 
                f"tot_in_rent: ${total_income_rental:,.2f}, " +
                f"tot_in_tot: ${total_taxable_income:,.2f}" 
                )

        total_tax *= scale
        print(f"\nTotal income tax: ${total_tax:,.2f} collected from {individuals.n_individuals} individuals")

        return total_tax

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
        """Return the taxation rate

        Returns:
            list[tuple[float, float]]: Taxation rate as recorded as (threshold, rate)
        """
        return self.brackets

    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Return the taxation rate
        
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
        """Return the taxation rate

        Returns:
            list[tuple[float, float]]: Taxation rate as recorded as (threshold, rate)
        """
        return self.brackets
    
    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Return the taxation rate
                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
        """
        self._validate_brackets(brackets)
        self.brackets = brackets

class PersonalIncomeTaxProtocol(Protocol):
    def compute_individual_tax(self, taxable_income: float) -> float: ...
    def get_rate(self) -> list[tuple[float, float]]: ...
    def set_rate(self, brackets: list[tuple[float, float]]) -> None: ...
