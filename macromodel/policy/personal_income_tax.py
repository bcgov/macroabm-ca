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
            nonrefundable_credit_timeseries (pd.DataFrame): Nonrefundable credit timeseries for use by country.update_credits()
            nonrefundable_credits (dict(PersonalIncomeTaxCredit)): Nonrefundable credit variables

        """
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year
        
        self.bracket_timeseries:pd.DataFrame = None
        self.nonrefundable_credit_timeseries:pd.DataFrame = None
        self.nonrefundable_credits = {}

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
    
    def compute_taxable_income(
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
        annual_refundable_credits = np.zeros_like(individuals.ts.current("employee_income"))
        annual_tax_owed_after_nonrefundable = np.zeros_like(individuals.ts.current("employee_income"))
        annual_tax_owed = np.zeros_like(individuals.ts.current("employee_income"))
        annual_tax_adjustment = np.zeros_like(individuals.ts.current("employee_income"))         

        for i in range(individuals.n_individuals):
            # calculate individual tax owed 
            # NOTE: individuals_taxable_income must be !UNSCALED! to align with values in self.brackets
            annual_tax_on_taxable_income[i] = self.compute_individual_tax(annual_taxable_income[i] / self.scale) * self.scale
            
            # calculate non-refundable credits received 
            # NOTE: MUST be done after annual_taxable_income is populated
            # NOTE: individuals_taxable_income must be !UNSCALED! to align with values in self.nonrefundable_credits
            annual_nonrefundable_credits[i] = self.compute_individual_nonrefundable_credits(
                            index=i,
                            annual_tax_on_taxable_income_unscaled = annual_tax_on_taxable_income[i] / self.scale,
                            annual_taxable_income_unscaled = annual_taxable_income / self.scale,    # pass whole array
                            individuals=individuals,
                            households=households,
                            ) * self.scale

            tax_minus_nonrefundable_credits = (
                annual_tax_on_taxable_income[i] 
                - annual_nonrefundable_credits[i]
            )
            # non-refundable credits check
            if  tax_minus_nonrefundable_credits < 0:
                annual_tax_owed_after_nonrefundable[i] = 0
            else:
                annual_tax_owed_after_nonrefundable[i] = tax_minus_nonrefundable_credits
            
            # calculate refundable credits received
            # NOTE: individuals_taxable_income must be !UNSCALED! to align with values in self.refundable_credits
            annual_refundable_credits[i] = self.compute_individual_refundable_credits(
                            index=i,
                            annual_tax_on_taxable_income_unscaled = annual_tax_on_taxable_income[i] / self.scale,
                            annual_taxable_income_unscaled = annual_taxable_income / self.scale,    # pass whole array
                            individuals=individuals,
                            households=households,
                            ) * self.scale
            
            # now apply refundable credits
            annual_tax_owed[i] = (
                            annual_tax_owed_after_nonrefundable[i] 
                            - annual_refundable_credits[i]
                        )
                        
            # calculat the difference between amount paid vs owed
            annual_tax_adjustment[i] = annual_tax_owed[i] - annual_personal_income_tax_paid[i]

        # store for this timestep
        individuals.ts.personal_income_tax_owed.append(annual_tax_owed / self.increments_per_year)
        individuals.ts.personal_income_tax_adjustment.append(annual_tax_adjustment / self.increments_per_year)

    def compute_individual_nonrefundable_credits(
            self,
            index: int,
            annual_tax_on_taxable_income_unscaled: float,
            annual_taxable_income_unscaled: np.ndarray,
            individuals: Individuals, 
            households: Households, 
            ) -> float:
        """Calculate personal income tax non-refundable credits owed to an indiviudal agent for a single timestep
            NOTE: Assumes that credits are applied consistently with Form BC428 2025 across time and jurisdiction
        
        Attributes:
            index (int): individual index
            annual_tax_on_taxable_income_unscaled (float): Individual tax on taxable income (unscaled)
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
                f"\nindex {index} ", 
                f"taxable_income {annual_taxable_income_unscaled[index]}"
            )

            hh_id = individuals.states["Corresponding Household ID"][index]
            hh_list = households.states["corr_individuals"][hh_id]
            print(f"hh list {hh_list}")
            print(f"hh age {individuals.states["Age"][hh_list]}")
            print(f"hh income {annual_taxable_income_unscaled[hh_list]}")

        # setup cost tracking for each credit
        tax_burden_remaining = annual_tax_on_taxable_income_unscaled
        # zero the costs for current credits
        credit_list = self.nonrefundable_credits
        for credit in credit_list:
            individuals.states["Credit Cost"][index][credit] = 0

        # NOTE: see Form BC428
        rate = 0

        # Basic personal amount
        if "Personal Amount" in self.nonrefundable_credits:
            basic_personal_amount = self.nonrefundable_credits["Personal Amount"].amount  # line 16
            rate = self.nonrefundable_credits["Personal Amount"].rate                     # line 55
            if rate == 0:
                raise ValueError("non-refundable tax credit rate cannot be zero")

            credit_cost_tuple = self.compute_nonrefundable_credit_cost(
                tax_burden_remaining = tax_burden_remaining,
                amount = basic_personal_amount,
                rate = rate,
                )
            individuals.states["Credit Cost"][index]["Personal Amount"], tax_burden_remaining = credit_cost_tuple
        else:
            basic_personal_amount = 0
            individuals.states["Credit Cost"][index]["Personal Amount"] = 0

        if debug: print(f"basic_personal_amount {basic_personal_amount}")

        # Age (65 or older by end of tax year) # NOTE: See Worksheet BC428
        if "Age Amount" in self.nonrefundable_credits:
            amount = self.nonrefundable_credits["Age Amount"].amount                      # line 1
            top = self.nonrefundable_credits["Age Amount"].top                            # line 58080 description
            clawback = self.nonrefundable_credits["Age Amount"].clawback                  # line 3
            clawback_rate = self.nonrefundable_credits["Age Amount"].clawback_rate        # line 5
            
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

            credit_cost_tuple = self.compute_nonrefundable_credit_cost(
                tax_burden_remaining = tax_burden_remaining,
                amount = age_amount,
                rate = rate,
                )
            individuals.states["Credit Cost"][index]["Age Amount"], tax_burden_remaining = credit_cost_tuple
        else:
            age_amount =  0
            individuals.states["Credit Cost"][index]["Age Amount"] = 0

        if debug: print(f"age_amount {age_amount}")

        # Spouse or common-law partner  # NOTE: see Form BC428
        if "Spousal Amount" in self.nonrefundable_credits:
            amount = self.nonrefundable_credits["Spousal Amount"].amount                   # line 20
            top = self.nonrefundable_credits["Spousal Amount"].top                         # line 18
            
            if individuals.states["Age"][index] >= 18:       # check if adult
                hh_id = individuals.states["Corresponding Household ID"][index]    # identify which hh it belongs to
                spousal_taxable_income = 0

                if households.states["Number of Adults"][hh_id] == 2:    # NOTE: assume all households with 2 adults are eligible
                    found_spouse = False
                    hh_list = households.states["corr_individuals"][hh_id]
                    for j in hh_list:
                        if j != index and individuals.states["Age"][j] >= 18:
                            spousal_taxable_income = annual_taxable_income_unscaled[j]
                            found_spouse = True
                    if found_spouse == False:
                        raise ValueError("Failed to find 2nd adult for households.states['Number of Adults'][hh_id] == 2")
                    
                    spousal_amount = top - spousal_taxable_income
                    if spousal_amount < 0:
                        spousal_amount = 0
                    elif spousal_amount > amount:
                        spousal_amount = amount
                else:
                    spousal_amount = 0 
            else:
                spousal_amount = 0

            credit_cost_tuple = self.compute_nonrefundable_credit_cost(
                tax_burden_remaining = tax_burden_remaining,
                amount = spousal_amount,
                rate = rate,
                )
            individuals.states["Credit Cost"][index]["Spousal Amount"], tax_burden_remaining = credit_cost_tuple
        else:
            spousal_amount = 0
            individuals.states["Credit Cost"][index]["Spousal Amount"] = 0

        if debug: print(f"spousal_amount {spousal_amount}")

        # Eligible dependant    # NOTE: see Form BC428
        if "Dependant Amount" in self.nonrefundable_credits:
            amount = self.nonrefundable_credits["Dependant Amount"].amount                # line 23
            top = self.nonrefundable_credits["Dependant Amount"].top                      # line 21

            if individuals.states["Age"][index] >= 18:       # check if adult
                hh_id = individuals.states["Corresponding Household ID"][index]    # identify which hh it belongs to

                if (households.states["Type"][hh_id] == HouseholdType.SINGLE_PARENT_WITH_CHILDREN):
                    dependant_amount = amount   # NOTE: children do not currently earn incomes in MacroABM
                else:
                    dependant_amount = 0
            else:
                dependant_amount = 0

            credit_cost_tuple = self.compute_nonrefundable_credit_cost(
                tax_burden_remaining = tax_burden_remaining,
                amount = dependant_amount,
                rate = rate,
                )
            individuals.states["Credit Cost"][index]["Dependant Amount"], tax_burden_remaining = credit_cost_tuple
        else:
            dependant_amount = 0
            individuals.states["Credit Cost"][index]["Dependant Amount"] = 0

        if debug: print(f"dependant_amount {dependant_amount}")

        if "Tax Reduction" in self.nonrefundable_credits:
            amount = self.nonrefundable_credits["Tax Reduction"].amount                # line 73
            top = self.nonrefundable_credits["Tax Reduction"].top                      # line 75
            clawback_rate = self.nonrefundable_credits["Tax Reduction"].clawback_rate  # line 77

            if annual_taxable_income_unscaled[index] > top:
                tax_reduction = 0
            else:
                clawback = (annual_taxable_income_unscaled[index] - top) * clawback_rate
                if clawback < 0:
                    clawback = 0
                tax_reduction = amount - clawback

            credit_cost_tuple = self.compute_nonrefundable_credit_cost(
                tax_burden_remaining = tax_burden_remaining,
                amount = tax_reduction,
                rate = 1,
                )
            individuals.states["Credit Cost"][index]["Tax Reduction"], tax_burden_remaining = credit_cost_tuple
        else:
            tax_reduction = 0   # for jurisdictions outside of BC
            individuals.states["Credit Cost"][index]["Tax Reduction"] = 0

        if debug: print(f"tax_reduction {tax_reduction}")

        # Donations and gifts amount (see Worksheet BC 428 2025 and supplemented by T1 Schedule 9)
        if "Donations Amount" in self.nonrefundable_credits:
            
            # NOTE: self.nonrefundable_credits parameter names (right side of equation) 
            #       do not map well to the meaning of the donation credit parameters (left side)
            minimum = self.nonrefundable_credits["Donations Amount"].amount                 # line 2
            top = self.nonrefundable_credits["Donations Amount"].top                        # line 5
            percentage_B = self.nonrefundable_credits["Donations Amount"].rate              # line 7
            percentage_C = self.nonrefundable_credits["Donations Amount"].clawback          # line 8
            percentage_D = self.nonrefundable_credits["Donations Amount"].clawback_rate     # line 9

            total_eligible_donations = individuals.states["Total Eligible Donations"][index] / self.scale  # already annual value                           
            total_eligible_donations_above_min = (
                total_eligible_donations 
                - min(total_eligible_donations, minimum)
            )                                                           # line 2
            if debug: print(f"total_eligible_donations_above_min {total_eligible_donations_above_min}")
            
            if annual_taxable_income_unscaled[index] - top < 0:         # line 6
                annual_taxable_income_unscaled_above_threshold = 0
            else:
                annual_taxable_income_unscaled_above_threshold = annual_taxable_income_unscaled[index] - top
            if debug: print(f"annual_taxable_income_unscaled_above_threshold {annual_taxable_income_unscaled_above_threshold}")

            A = total_eligible_donations_above_min
            B = min(total_eligible_donations_above_min, annual_taxable_income_unscaled_above_threshold)     # line 7
            C = A - B                                                                                       # line 8 
            D = min(total_eligible_donations, minimum)                                                      # line 9
            if debug: 
                print(f"B {B} B * percentage_B {B * percentage_B}") 
                print(f"C {C} C * percentage_C {C * percentage_C}")
                print(f"D {D} D * percentage_D {D * percentage_D}") 

            donation_amount = (B * percentage_B) + (C * percentage_C) + (D * percentage_D)
        else:
            donation_amount = 0

        if debug: print(f"donation_amount {donation_amount}")

        total_individual_credits = (
            basic_personal_amount + age_amount + spousal_amount + dependant_amount) * rate  # line 56
        total_individual_credits += donation_amount                                         # line 57
        total_individual_credits += tax_reduction                                           # line 80

        if debug: print(f"total_credits {total_individual_credits}")

        # store credit costs    # temporary TODO: fix for dynamic number of credits
        individuals.ts.current("personal_income_tax_credit_cost_personal")[index] = (
            individuals.states["Credit Cost"][index]["Personal Amount"]
        )
        individuals.ts.current("personal_income_tax_credit_cost_age")[index] = (
                    individuals.states["Credit Cost"][index]["Age Amount"]
                )
        individuals.ts.current("personal_income_tax_credit_cost_spouse")[index] = (
            individuals.states["Credit Cost"][index]["Spousal Amount"]
        )
        individuals.ts.current("personal_income_tax_credit_cost_dependant")[index] = (
            individuals.states["Credit Cost"][index]["Dependant Amount"]
        )
        individuals.ts.current("personal_income_tax_credit_cost_reduction")[index] = (
            individuals.states["Credit Cost"][index]["Tax Reduction"]
        )

        return total_individual_credits

    def compute_nonrefundable_credit_cost(
            self,
            tax_burden_remaining: float,
            amount: float,
            rate: float,
            ) -> tuple[float, float]:
        """Calculate cost of supplying individual with a nonrefundable tax credit
        
        Attributes:
            tax_burden_remaining (float):
            amount (float):
            rate (float):

        Returns:
            float: Cost of supplying individual with tax credit
            tuple[float, float]: 
                Cost of supplying individual with tax credit, 
                tax_burden_remaining
        """

        if tax_burden_remaining <= 0:                   # no tax burden to offset with further refunds
            tax_burden_remaining = 0
            credit_cost = 0
        else:                                           # tax burden available to offset with refunds
            if tax_burden_remaining >= amount * rate:   # whole credit can be applied
                tax_burden_remaining -= amount * rate
                credit_cost = amount * rate
            else:                                       # partial credit can be applied
                tax_burden_remaining = 0
                credit_cost = tax_burden_remaining

        return credit_cost, tax_burden_remaining

    def compute_individual_refundable_credits(
            self,
            index: int,
            annual_tax_on_taxable_income_unscaled: float,
            annual_taxable_income_unscaled: np.ndarray,
            individuals: Individuals, 
            households: Households, 
            ) -> float:
        """Calculate personal income tax refundable credits owed to an indiviudal agent for a single timestep
            NOTE: Assumes that credits are applied consistently with Form BC479 2025 across time and jurisdiction
        
        Attributes:
            index (int): individual index
            annual_tax_on_taxable_income_unscaled (float): Individual tax on taxable income (unscaled)
            annual_taxable_income_unscaled (np.ndarray): Annual individual taxable incomes (unscaled)
            individuals (Individuals): agent    TODO replace with np.array for each attribute used
            households (Households): agent      TODO replace with np.array for each attribute used

        Returns:
            float: Personal income tax refundable credits owed to an individual agent
        """
        from macromodel.agents.households.household_properties import HouseholdType

        debug = False

        if debug:
            hh_id = individuals.states["Corresponding Household ID"][index]
            hh_list = households.states["corr_individuals"][hh_id]

            if households.states["Tenure Status of the Main Residence"][hh_id] == 3:
                print(
                    f"\nindex {index} ",
                    f"hh_id {hh_id} ",
                    f"Tenure Status {households.states["Tenure Status of the Main Residence"][hh_id]} ",  
                    f"taxable_income {annual_taxable_income_unscaled[index]}"
                )
                print(f"hh list {hh_list}")
                print(f"hh age {individuals.states["Age"][hh_list]}")
                print(f"hh income {annual_taxable_income_unscaled[hh_list]}")

        # TODO: implement data readers etc before uncommenting (https://fincsp.atlassian.net/browse/FREDA-352)
        # if "Renter Amount" in self.refundable_credits:
        #     amount = self.refundable_credits["Renter Amount"].amount
        #     top = self.refundable_credits["Renter Amount"].top
        
        amount = 400                                                # line 38
        top = 84764                                                 # credit description
        clawback = 64764                                            # line 40
        clawback_rate = 0.02                                        # line 42

        if (individuals.states["Age"][index] >= 18):       # check if adult
            hh_id = individuals.states["Corresponding Household ID"][index]    # identify which hh it belongs to
            hh_list = households.states["corr_individuals"][hh_id]

            # NOTE: this doesn't include rental or investment income
            hh_net_income_unscaled = np.sum(annual_taxable_income_unscaled[hh_list])

            # filter for renters (see housing_info_from_population() in matching_households_with_houses.py 
            # for "Tenure Status of the Main Residence" meaning)
            if (
                households.states["Tenure Status of the Main Residence"][hh_id] == 3
                and hh_net_income_unscaled < top
            ):
            
                # determine which adult in hh claims credit (only one allowed)
                if households.states["Number of Adults"][hh_id] > 1:
                    hh_annual_taxable_income_unscaled = annual_taxable_income_unscaled[hh_list]

                    # Sort pairs by individual incomes within hh
                    paired = list(zip(hh_list, hh_annual_taxable_income_unscaled))
                    paired.sort(key=lambda x: x[1])
                    hh_list_sorted, hh_annual_taxable_income_unscaled_sorted = zip(*paired)
                    hh_list_sorted = list(hh_list_sorted)
                    hh_annual_taxable_income_unscaled_sorted = list(hh_annual_taxable_income_unscaled_sorted)

                    # assign credit to adult individual with highest income
                    if index == hh_list_sorted[-1]:
                        if debug:
                            print(f"hh_list_sorted {hh_list_sorted}")
                            print(f"hh_annual_taxable_income_unscaled_sorted {hh_annual_taxable_income_unscaled_sorted}")
                        assigned_credit = True
                    else:
                        assigned_credit = False
                else:
                    assigned_credit = True

                if assigned_credit:
                    if hh_net_income_unscaled <= clawback:
                        renter_amount =  amount
                    else:
                        renter_amount = (
                            amount - 
                            (hh_net_income_unscaled - clawback) 
                            * clawback_rate
                        )
                else:
                    renter_amount = 0
            else:
                renter_amount = 0
        else:
            renter_amount = 0

        if debug:
            print(f"renter_amount {renter_amount}")

        refundable_credit_amount = renter_amount

        return refundable_credit_amount
    

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
            nonrefundable_credit_timeseries (pd.DataFrame): Nonrefundable credit timeseries for use by country.update_credits()
            nonrefundable_credits (dict(PersonalIncomeTaxCredit)): Nonrefundable credit variables

        """
        self._validate_brackets(brackets)
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year

        self.bracket_timeseries:pd.DataFrame = None
        self.nonrefundable_credit_timeseries:pd.DataFrame = None
        self.nonrefundable_credits = {}

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
            nonrefundable_credit_timeseries (pd.DataFrame): Nonrefundable credit timeseries for use by country.update_credits()
            nonrefundable_credits (dict(PersonalIncomeTaxCredit)): Nonrefundable credit variables 
        """

        self._validate_brackets(brackets)
        self.brackets = brackets
        self.scale = scale
        self.increments_per_year = increments_per_year

        self.bracket_timeseries:pd.DataFrame = None
        self.nonrefundable_credit_timeseries:pd.DataFrame = None
        self.nonrefundable_credits = {}

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
