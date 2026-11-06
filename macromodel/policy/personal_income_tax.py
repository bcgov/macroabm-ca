from abc import ABC, abstractmethod
from typing import Protocol
from macromodel.timestep import Timestep
from macromodel.agents.households.households import Households
from macromodel.agents.individuals.individuals import Individuals
from macromodel.agents.individuals.individual_properties import ActivityStatus
import numpy as np
import pandas as pd 
# import inspect

class PersonalIncomeTaxCredit():
    """Personal income tax credit policy."""
    def __init__(self, amount: float, top: float, rate:float, clawback: float, clawback_rate:float, indexed: int):
        """Initialize a personal income tax credit policy.
        
        Attributes:
            amount (float): Base credit amount
            top (float): Maximum annual income for eligibility
            rate (float): Credit rate
            clawback (float): Annual income at which clawback begins
            clawback_rate (float): Clawback rate
            indexed (int): Toggle whether values are indexed to inflation

        """
        self.amount = amount
        self.top = top
        self.rate = rate
        self.clawback = clawback
        self.clawback_rate = clawback_rate
        self.indexed = indexed

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
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (upper income threshold, rate)
            scale (int): Scaling factor for population-based calculations
            increments_per_year (int): Timesteps per year
            bracket_timeseries (pd.DataFrame): Bracket timeseries for use by country.update_brackets()
            credits_timeseries (pd.DataFrame): Credit timeseries for use by country.update_credits()
            credits (dict(PersonalIncomeTaxCredit)): Credit variables

        """
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year
        
        self.bracket_timeseries:pd.DataFrame = None
        self.credit_timeseries:pd.DataFrame = None
        self.credits = {}

    @abstractmethod
    def compute_individual_tax(self, taxable_income: float) -> float:
        """Calculate individual tax based on income."""
        pass

    @abstractmethod
    def set_brackets(self, brackets: list[tuple[float, float]]) -> None:
        """Set the taxation brackets for this timestep"""
        pass

    def get_brackets(self) -> list[tuple[float, float]]:
        """Return the taxation rate for this timestep
        
        Returns:
            list[tuple[float, float]]: Taxation rate as recorded as (upper income threshold, rate)
        """
        return self.brackets

    def compute_withhold(
            self,
            income_component_by_individual: np.ndarray, 
            ) -> np.ndarray:
        """Calculate estimate of income to be withheld to pay personal income tax (without credits) for one timestep
        
        Attributes:
            income_component_by_individual (np.ndarray): Gross income (ie real wages or unemployment benefits)

        Returns:
            np.ndarray: Estimate of personal income tax owed for a single timestep
        """

        # blanks for calculation
        individuals_personal_income_tax_owed = np.zeros_like(income_component_by_individual)

        # estimated component of personal income tax owed (without credits)
        for i in range(len(income_component_by_individual)):
            # NOTE: individuals income must be !UNSCALED! to align with values in self.brackets
            projected_annual_tax= self.compute_individual_tax(
                income_component_by_individual[i]/self.scale * self.increments_per_year
                ) * self.scale
            individuals_personal_income_tax_owed[i] = projected_annual_tax / self.increments_per_year

        return individuals_personal_income_tax_owed 
    
    def new_compute_taxable_income(
            self, 
            individuals: Individuals, 
            households: Households, 
            ) -> None:
        """Calculate taxable income for all indiviudal agents for one timestep
        
        Attributes:
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used
        """                   

        debug_individuals = False
        debug_total_components = False

        # blanks for calculation
        individuals_taxable_income = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_employment = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_unemployment = np.zeros_like(individuals.ts.current("employee_income"))
        individuals_taxable_income_rental = np.zeros_like(individuals.ts.current("employee_income"))
        # individuals_taxable_income_investment = np.zeros_like(individuals.ts.current("employee_income"))  # TODO: implement

        # calculate all taxable income streams across individuals
        for i in range(individuals.n_individuals):
            if individuals.ts.current("employee_income_gross")[i] < 0:
                raise ValueError("employee_income cannot be negative.")
            # TODO: future extension will mean individuals_taxable_income_employment != employee_income_gross
            #       this will come with a feature to convert total income -> net income -> taxable income
            individuals_taxable_income_employment[i] = individuals.ts.current("employee_income_gross")[i]   

            if individuals.ts.current("income_from_unemployment_benefits")[i] < 0:
                raise ValueError("income_from_unemployment_benefits cannot be negative.")
            individuals_taxable_income_unemployment[i] = individuals.ts.current("income_from_unemployment_benefits")[i]

            # caclulate rental income (assuming it is split between all adult residents)
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

            if debug_individuals:    # !UNSCALED! diagnostic so that it can be compared to real world individuals
                print(
                    f"{i}: " +
                    f"hh_id: {individuals.states["Corresponding Household ID"][i]}, " +
                    f"employed: {individuals.states["Activity Status"][i] == ActivityStatus.EMPLOYED}, " +
                    f"income_employment_unscaled_annual: ${individuals_taxable_income_employment[i]/self.scale * self.increments_per_year:,.2f}, " + 
                    f"income_unemployment_unscaled_annual: ${individuals_taxable_income_unemployment[i]/self.scale * self.increments_per_year:,.2f}, " + 
                    f"income_rental_unscaled_annual: ${individuals_taxable_income_rental[i]/self.scale * self.increments_per_year:,.2f}, " +
                    f"taxable_income_unscaled_annual: ${individuals_taxable_income[i]/self.scale * self.increments_per_year:,.2f} "
                    )
                                        
        if debug_total_components:  # !SCALED! diagnostic so that it can be compared to provincial values
            print(
                f"total income_employment: ${np.sum(individuals_taxable_income_employment):,.2f}, " + 
                f"total income_unemployment: ${np.sum(individuals_taxable_income_unemployment):,.2f}, " + 
                f"total income_rental: ${np.sum(individuals_taxable_income_rental):,.2f}, " +
                f"total taxable_income: ${np.sum(individuals_taxable_income):,.2f}" 
            )

        # store income streams
        individuals.ts.taxable_income.append(individuals_taxable_income)
        individuals.ts.taxable_income_employment.append(individuals_taxable_income_employment)
        individuals.ts.taxable_income_unemployment.append(individuals_taxable_income_unemployment)
        individuals.ts.taxable_income_rental.append(individuals_taxable_income_rental)

    def compute_personal_income_tax_owed(
            self, 
            annual_taxable_income: np.ndarray,
            annual_personal_income_tax_paid: np.ndarray,
            individuals: Individuals, 
            households: Households,
            ) -> None:
        """Calculate personal income tax owed for all indiviudal agents and reconcile with amounts paid
        
        Attributes:
            annual_taxable_income (np.ndarray): Projected (or actual) annual individual taxable incomes
            annual_personal_income_tax_paid (np.ndarray): Projected (or actual) annual individual personal income tax paid
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used
        """                   

        # blanks for calculation
        annual_tax_on_taxable_income = np.zeros_like(individuals.ts.current("employee_income"))
        annual_nonrefundable_credits = np.zeros_like(individuals.ts.current("employee_income"))
        annual_tax_owed = np.zeros_like(individuals.ts.current("employee_income"))
        annual_tax_adjustment = np.zeros_like(individuals.ts.current("employee_income"))         

        for i in range(individuals.n_individuals):
            # calculate individual tax owed 
            # NOTE: individuals_taxable_income must be !UNSCALED! to align with values in self.brackets
            annual_tax_on_taxable_income[i] = self.compute_individual_tax(annual_taxable_income[i] / self.scale) * self.scale
            
            # calculate non-refundable credits received 
            # NOTE: MUST be done after annual_taxable_income is populated
            # NOTE: individuals_taxable_income must be !UNSCALED! to align with values in self.brackets
            annual_nonrefundable_credits[i] = self.compute_individual_nonrefundable_credits(
                            index=i,
                            annual_taxable_income_unscaled = annual_taxable_income / self.scale,    # pass whole array
                            individuals=individuals,
                            households=households,
                            ) * self.scale
            
            # projected_annual_tax_reduction = self.compute_individual_tax_reduction()  # TODO: implement
            tax_minus_credits = (
                annual_tax_on_taxable_income[i] 
                - annual_nonrefundable_credits[i]
                # - projected_annual_tax_reduction                                      # TODO: implement
            )
            if  tax_minus_credits < 0:
                annual_tax_owed[i] = 0
            else:
                annual_tax_owed[i] = tax_minus_credits
                        
            # calculat the difference between amount paid vs owed
            annual_tax_adjustment[i] = annual_tax_owed[i] - annual_personal_income_tax_paid[i]

        # store for this timestep
        individuals.ts.personal_income_tax_owed.append(annual_tax_owed / self.increments_per_year)
        individuals.ts.personal_income_tax_adjustment.append(annual_tax_adjustment / self.increments_per_year)

    def compute_individual_nonrefundable_credits(
            self,
            index: int,
            annual_taxable_income_unscaled: np.ndarray,
            individuals: Individuals, 
            households: Households, 
            ) -> float:
        """Calculate personal income tax non-refundable credits owed to an indiviudal agent for a single timestep
            NOTE: Assumes that credits are applied consistently with Form BC428 2025 across time and jurisdiction
        
        Attributes:
            index (int): individual index
            annual_taxable_income_unscaled (np.ndarray): Annual individual taxable incomes (unscaled)
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used

        Returns:
            float: Personal income tax non-refundable credits owed to an individual agent
        """
        from macromodel.agents.households.household_properties import HouseholdType

        debug = False

        if debug:
            print(
                f"index {index} ", 
                f"taxable_income {annual_taxable_income_unscaled[index]}"
            )

            hh_id = individuals.states["Corresponding Household ID"][index]
            hh_list = households.states["corr_individuals"][hh_id]
            print(f"hh list {hh_list}")
            print(f"hh age {individuals.states["Age"][hh_list]}")
            print(f"hh income {annual_taxable_income_unscaled[hh_list]}")

        # NOTE: see Form BC428
        rate = 0

        # Basic personal amount
        if "Personal Amount" in self.credits:
            basic_personal_amount = self.credits["Personal Amount"].amount  # line 16
            rate = self.credits["Personal Amount"].rate                     # line 55
            if rate == 0:
                raise ValueError("non-refundable tax credit rate cannot be zero")
        else:
            basic_personal_amount = 0
        if debug: print(f"basic_personal_amount {basic_personal_amount}")

        # Age (65 or older by end of tax year) # NOTE: See Worksheet BC428
        if "Age Amount" in self.credits:
            amount = self.credits["Age Amount"].amount                      # line 1
            top = self.credits["Age Amount"].top                            # line 58080 description
            clawback = self.credits["Age Amount"].clawback                  # line 3
            clawback_rate = self.credits["Age Amount"].clawback_rate        # line 5
            
            if individuals.states["Age"][index] >= 65:
                if annual_taxable_income_unscaled[index] <= clawback:
                    age_amount =  amount
                elif annual_taxable_income_unscaled[index] > top:
                    age_amount = 0
                else:
                    age_amount = (
                        amount - 
                        (annual_taxable_income_unscaled[index] - clawback) 
                        * clawback_rate
                    )
            else:
                age_amount =  0
        else:
            age_amount =  0
        if debug: print(f"age_amount {age_amount}")

        # Spouse or common-law partner  # NOTE: see Form BC428
        if "Spouse Amount" in self.credits:
            amount = self.credits["Spouse Amount"].amount                   # line 20
            top = self.credits["Spouse Amount"].top                         # line 18
            
            if (
                individuals.states["Corresponding Household ID"][index] != 0
                and individuals.states["Age"][index] >= 18
                ):       # check if adult and belongs to a hh
                hh_id = individuals.states["Corresponding Household ID"][index]    # identify which hh it belongs to

                if households.states["Number of Adults"][hh_id] == 2:    # NOTE: assume all households with 2 adults are eligible
                    found_spouse = False
                    hh_list = households.states["corr_individuals"][hh_id]
                    for j in hh_list:
                        if j != index and individuals.states["Age"][j] >= 18:
                            spouse_taxable_income = annual_taxable_income_unscaled[j]
                            found_spouse = True
                    if found_spouse == False:
                        raise ValueError("Failed to find 2nd adult for households.states['Number of Adults'][hh_id] == 2")
                    
                    spouse_amount = top - spouse_taxable_income
                    if spouse_amount < 0:
                        spouse_amount = 0
                    elif spouse_amount > amount:
                        spouse_amount = amount
                else:
                    spouse_amount = 0 
            else:
                spouse_amount = 0
        else:
            spouse_amount = 0
        if debug: print(f"spouse_amount {spouse_amount}")

        # Eligible dependant    # NOTE: see Form BC428
        if "Dependant Amount" in self.credits:
            amount = self.credits["Dependant Amount"].amount                # line 23
            top = self.credits["Dependant Amount"].top                      # line 21

            if (
                individuals.states["Corresponding Household ID"][index] != 0
                and individuals.states["Age"][index] >= 18
            ):       # check if adult and belongs to a hh
                hh_id = individuals.states["Corresponding Household ID"][index]    # identify which hh it belongs to

                if (households.states["Type"][hh_id] == HouseholdType.SINGLE_PARENT_WITH_CHILDREN):
                    dependant_amount = amount   # NOTE: children do not currently earn incomes in MacroABM
                else:
                    dependant_amount = 0
            else:
                dependant_amount = 0
        else:
            dependant_amount = 0

        if debug: print(f"dependant_amount {dependant_amount}")

        tax_reduction = 0   # for jurisdictions outside of BC
        if "Tax Reduction" in self.credits:
            amount = self.credits["Tax Reduction"].amount                # line 73
            top = self.credits["Tax Reduction"].top                      # line 75
            clawback_rate = self.credits["Tax Reduction"].clawback_rate  # line 77

            if annual_taxable_income_unscaled[index] > top:
                tax_reduction = 0
            else:
                clawback = (annual_taxable_income_unscaled[index] - top) * clawback_rate
                if clawback < 0:
                    clawback = 0
                tax_reduction = amount - clawback

        total_individual_credits = (basic_personal_amount + age_amount + spouse_amount + dependant_amount) * rate # line 56
        total_individual_credits += tax_reduction                       # line 80

        if debug: print(f"total_credits {total_individual_credits}")       

        return total_individual_credits 

class FlatRate(PersonalIncomeTax):
    """Flat rate income tax policy."""
    def __init__(self, brackets: list[tuple[float, float]], scale: int, increments_per_year: int):
        """Initialize Flat rate income tax policy.
        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (upper income threshold, rate)
                Example [(float('inf'), 0.15)]
                Only threshold must be float('inf')
            scale (int): Scaling factor for population-based calculations
            increments_per_year (int): Timesteps per year
            bracket_timeseries (pd.DataFrame): Bracket timeseries for use by country.update_brackets()
            credits_timeseries (pd.DataFrame): Credit timeseries for use by country.update_credits()
            credits (dict(PersonalIncomeTaxCredit)): Credit variables

        """
        self._validate_brackets(brackets)
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year

        self.bracket_timeseries:pd.DataFrame = None
        self.credit_timeseries:pd.DataFrame = None
        self.credits = {}

    def _validate_brackets(self, brackets: list[tuple[float, float]]) -> None:
        """Ensure brackets are valid
                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (upper income threshold, rate)
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
            taxable_income (float): Unscaled taxable income

        Returns:
            float: Personal income tax owed by an individual agent
        """
        if taxable_income < 0:
            raise ValueError("Taxable income cannot be negative.")
        _, rate = self.brackets[0]

        return taxable_income * rate

    def set_brackets(self, brackets: list[tuple[float, float]]) -> None:
        """Set the taxation brackets for this timestep
        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (upper income threshold, rate)
        """
        self._validate_brackets(brackets)        
        self.brackets = brackets

class ProgressiveRate(PersonalIncomeTax):
    """Progressive rate income tax policy with brackets."""
    def __init__(self, brackets: list[tuple[float, float]], scale: int, increments_per_year: int):
        """Initialize Progressive rate income tax policy.
                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (upper income threshold, rate)
                Example [(50000, 0.1), (100000, 0.2), (float('inf'), 0.3)] 
                Final threshold must be float('inf')
            scale (int): Scaling factor for population-based calculations
            increments_per_year (int): Timesteps per year
            bracket_timeseries (pd.DataFrame): Bracket timeseries for use by country.update_brackets()
            credits_timeseries (pd.DataFrame): Credit timeseries for use by country.update_credits()
            credits (dict(PersonalIncomeTaxCredit)): Credit variables 
        """

        self._validate_brackets(brackets)
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year

        self.bracket_timeseries:pd.DataFrame = None
        self.credit_timeseries:pd.DataFrame = None
        self.credits = {}

    def _validate_brackets(self, brackets: list[tuple[float, float]]) -> None:
        """Ensure brackets are valid
                        
        Attributes:
            brackets (list[tuple[float, float]]): Taxation rate as recorded as (upper income threshold, rate)
        """
        if not brackets:
            raise ValueError("ProgressiveRate brackets cannot be empty.")
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
            taxable_income (float): Unscaled taxable income

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
    
    def set_brackets(self, brackets: list[tuple[float, float]]) -> None:
        """Set the taxation brackets for this timestep
                
        Attributes:
            brackets (list[tuple[float, float]]): Taxation brackets as recorded as (upper income threshold, rate)
        """
        self._validate_brackets(brackets)
        self.brackets = brackets

class PersonalIncomeTaxProtocol(Protocol):
    def compute_individual_tax(self, taxable_income: float) -> float: ...
    def get_brackets(self) -> list[tuple[float, float]]: ...
    def set_brackets(self, brackets: list[tuple[float, float]]) -> None: ...
