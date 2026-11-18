from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor 
from typing import List, Tuple, Union
# import inspect

class StateAugmentation():
    """Individual state augementation"""
    def __init__(
            self, 
            name: str, 
            data: pd.DataFrame, 
            # age_brackets: list[tuple[float, float]], 
            # income_brackets: list[tuple[float, float]], 
            # state_mu: np.ndarray, 
            # state_sigma: np.ndarray,
        ):
        """Initialize a state augmentation.
        
        Attributes:
            name (str): Augmentation identifier
            data (Optional[pd.DataFrame]): Individual data for estimating a state based on age and income
            # age_brackets (list[tuple[float, float]]): Age brackets for estimating state amounts 
            #                                              in the form (lower threshold, upper threshold)
            # income_brackets (list[tuple[float, float]]): Income brackets for estimating state amounts 
            #                                              in the form (lower threshold, upper threshold)
            # state_mu (Optional[np.ndarray)]: Mean value for a given [age, income]
            # state_sigma (Optional[np.ndarray]): sqrt(variance) value for a given [age, income]

        """
        # print("START: StateAugmentation.__init__()")
        debug = False

        self.name = name
        self.data = data

        # Distributional regression model parameters
        n_estimators=300
        max_depth=3
        learning_rate=0.05
        random_state=42
        data_reset_index = data.reset_index()
        X = data_reset_index[["age", "income"]]
        y = data_reset_index["amount"]

        # Compute mu model
        mu_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        mu_model.fit(X, y)

        mu_hat = mu_model.predict(X)
        residuals = y - mu_hat
        variance_target = residuals**2

        # Compute sigma model
        sigma_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        sigma_model.fit(X, variance_target)

        # build state brackets
        age_brackets_step = 10
        self.age_brackets = self.generate_float_brackets(start=0, bracket_step=age_brackets_step, num_brackets=7)

        income_brackets_step = 20000
        self.income_brackets = self.generate_float_brackets(start=0, bracket_step=income_brackets_step, num_brackets=7)

        # population state_mu and state_sigma
        state_mu = np.empty((len(self.age_brackets),len(self.income_brackets)))
        state_sigma = np.empty((len(self.age_brackets),len(self.income_brackets)))
        for m in range(len(self.age_brackets)):
            if debug: print("\n")
            for n in range(len(self.income_brackets)):

                # print(f"{m} age {age_brackets[m]} {n} income {income_brackets[n]}")

                if self.age_brackets[m][1] != float('inf'):
                    mid_age = self.age_brackets[m][0] + (self.age_brackets[m][1] - self.age_brackets[m][0])/2
                else:   # TODO: Different assumptions for this condition should be tested
                    mid_age = self.age_brackets[m][0] + age_brackets_step/2

                if self.income_brackets[n][1] != float('inf'):
                    mid_income = self.income_brackets[n][0] + (self.income_brackets[n][1] - self.income_brackets[n][0])/2
                else:   # TODO: Different assumptions for this condition should be tested
                    mid_income = self.income_brackets[n][0] + income_brackets_step/2

                features = pd.DataFrame({
                    "age": [mid_age],
                    "income": [mid_income]
                })
                
                mu = mu_model.predict(features)[0]
                variance = sigma_model.predict(features)[0]
                variance = max(variance, 1e-6)
                sigma = np.sqrt(variance)

                state_mu[m][n] = mu
                state_sigma[m][n] = sigma

                if debug: print(f"age {mid_age:,.2f} income ${mid_income:,.2f} \tmu ${state_mu[m][n]:,.2f} sigma {state_sigma[m][n]:.3e}")

        self.state_mu = state_mu
        self.state_sigma = state_sigma

        # print("END: StateAugmentation.__init__()")

    def generate_float_brackets(
            self,
            start: float,
            bracket_step: float,
            num_brackets: int = None,
            max_value: Union[float, None] = None
        ) -> List[Tuple[float, float]]:
        """
        Generate income brackets.
        
        Args:
            start (float): Starting income value
            bracket_step (float): Step size of each bracket
            num_brackets (int, optional): Number of brackets before the last one
            max_value (float, optional): Maximum value for the last bracket
        
        Returns:
            List[Tuple[float, float]]: List of (lower_bound, upper_bound) tuples.
        """
        # Input validation
        if bracket_step <= 0:
            raise ValueError("Bracket size must be positive.")
        if start < 0:
            raise ValueError("Start value must be non-negative.")
        if num_brackets is None and max_value is None:
            raise ValueError("Either num_brackets or max_value must be provided.")
        if num_brackets is not None and num_brackets <= 0:
            raise ValueError("Number of brackets must be positive.")

        brackets = []
        current_start = start

        # Determine number of brackets if max_value is given
        if num_brackets is None:
            num_brackets = int((max_value - start) // bracket_step)

        for i in range(num_brackets):
            current_end = current_start + bracket_step - 0.01  # avoid overlap
            brackets.append((current_start, current_end))
            current_start += bracket_step

        # Add last bracket
        if max_value is not None:
            brackets.append((current_start, max_value))
        else:
            brackets.append((current_start, float('inf')))

        return brackets

    def update_states_from_data(
            self,
            age: np.ndarray,
            income_unscaled: np.ndarray,
            scale: int,
            increments_per_year: int,
        ) -> np.ndarray:
        """
        Update individual states that rely on FIN data to the current timestep.
        
        Args:
            age (np.ndarray): Individual age
            income_unscaled (np.ndarray): Individual income (unscaled)
            scale (int): Population scaling factor
            increments_per_year (int): Timesteps per year
        
        Returns:
            np.ndarray: Updated individual states
        """
        
        # TODO: update to recalculate StateAugmentation.mu AND StateAugmentation.sigma for current year
        #       (doing so would put new computation burden on each timestep)

        debug_individual = False
        debug_overall = False

        # retrieve estimated states for each individual agent
        assigned_value = np.zeros(len(age))
        for i in range(len(age)):
            # determine income range of this individual
            ind_income_bracket_index = None
            ind_income_bracket_range = None
            for idx, (low, high) in enumerate(self.income_brackets, start=1):
                if low <= income_unscaled[i] <= high:
                    ind_income_bracket_index = idx - 1
                    ind_income_bracket_range = (low, high)
                    break
            if debug_individual: print(
                f"{i} income ${income_unscaled[i]:,.2f} "
                f"\tbracket_num {ind_income_bracket_index} "
                f"bracket_range {ind_income_bracket_range}"
            )

            # determine age range of this individual
            ind_age_bracket_index = None
            ind_age_bracket_range = None
            for idx, (low, high) in enumerate(self.age_brackets, start=1):
                if low <= age[i] <= high:
                    ind_age_bracket_index = idx - 1
                    ind_age_bracket_range = (low, high)
                    break
            if debug_individual: print(
                f"{i} age {age[i]:,.2f} "
                f"\tbracket_num {ind_age_bracket_index} "
                f"bracket_range {ind_age_bracket_range}"
            )

            # calculate state value             # NOTE: Must be !SCALED! quarterly amount to interact with model
            assigned_value[i] = np.random.normal(
                self.state_mu[ind_age_bracket_index][ind_income_bracket_index], 
                self.state_sigma[ind_age_bracket_index][ind_income_bracket_index],
                ) * scale / increments_per_year

            if assigned_value[i] < 0:                   # ensure amount is positive independent of estimation
                assigned_value[i] = 0

            if debug_overall:                           # test values 
                ind_indices = [5,10,500,1000,1500,2000,2500]
                if i in ind_indices:
                    print(
                        f"{i} \tage {age[i]} "
                        f"income ${income_unscaled[i]:,.2f} "
                        f"amount ${assigned_value[i] / scale * increments_per_year:,.2f}"
                    )

        if debug_overall:                               # test for max income
            max_income = np.max(income_unscaled)    
            i = np.where(income_unscaled == max_income)
            i = i[0]
            i = i[0]
            print(
                f"{i} \tage {age[i]} "
                f"income ${income_unscaled[i]:,.2f} "
                f"amount ${assigned_value[i] / scale * increments_per_year:,.2f}"
            )

        return assigned_value