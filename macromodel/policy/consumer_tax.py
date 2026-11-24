from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
# import inspect

class ConsumerTax():
    """Consumer tax credit policy."""
    def __init__(
            self, 
            rates: np.ndarray,
            scale: int, 
            increments_per_year: int ,
        ):
        """Initialize a consumer tax credit policy.
        
        Attributes:
            rates (np.ndarray): Taxation rate for each industry
            scale (int): Scaling factor for population-based calculations
            increments_per_year (int): Timesteps per year
            rate_timeseries (pd.DataFrame): Rate timeseries for use by country.update_brackets()
        """

        # self._validate_rates(rates)     # TODO: create self._validate_rates()
        self.rates = rates
        self.scale = scale
        self.increments_per_year = increments_per_year
        self.rates_timeseries:pd.DataFrame = None

    def get_rates(self) -> np.ndarray:
        """Return the taxation rates for this timestep
        
        Returns:
            np.ndarray: Taxation rates for each industry
        """
        return self.rates
    
    def set_rates(self, rates: np.ndarray) -> None:
        """Set the taxation rates for this timestep
                
        Attributes:
            rate (np.ndarray]): Taxation rates for each industry
        """
        self.rates = rates