"""Individual time series data management.

This module handles the creation and tracking of time series data for individuals
in the macroeconomic model. It captures:
- Income streams and distributions
- Labor market participation
- Wage expectations
- Unemployment benefits

The time series track both individual-level data and aggregate distributions
to analyze:
- Income inequality
- Labor market dynamics
- Social welfare outcomes
- Economic mobility
"""

import pandas as pd
import numpy as np

from typing import Any
from enum import Enum

from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


def create_individuals_timeseries(data: pd.DataFrame, scale: int) -> TimeSeries:
    """Create time series tracking for individual economic variables.

    This function initializes time series objects to track individual-level
    economic outcomes and their distributions over time. It captures:
    - Employment income
    - Unemployment benefits
    - Total income
    - Labor market participation
    - Wage expectations

    The function creates both individual-level series and histograms
    for distributional analysis.

    Args:
        data (pd.DataFrame): Individual-level data containing economic variables
        scale (int): Scale factor for histogram binning

    Returns:
        TimeSeries: Time series object containing:
            - n_individuals: Total number of individuals
            - employee_income: Employment-based income by individual
            - employee_income_histogram: Distribution of employment income
            - income_from_unemployment_benefits: Unemployment benefits by individual
            - income: Total income by individual
            - expected_income: Expected future income by individual
            - income_histogram: Distribution of total income
            - labour_inputs: Labor market participation by individual
            - reservation_wages: Minimum acceptable wages by individual
            - employee_income_gross: Pre-tax employment income
            - taxable_income: Taxable income from all income streams
            - taxable_income_employment: Taxable income from employment income
            - taxable_income_unemployment: Taxable income from unemployment income
            - taxable_income_rental: Taxable income from rental income
            - taxable_income_investment: Taxable income from investment income
            - personal_income_tax_owed: Personal income tax owed for all income streams
            - personal_income_tax_paid: Personal income tax paid for all income streams
            - personal_income_tax_paid_employment: Personal income tax paid for employment income
    """
    return TimeSeries(
        n_individuals=len(data),
        #
        employee_income=data["Employee Income"].values,
        employee_income_histogram=get_histogram(data["Employee Income"].values, scale),
        income_from_unemployment_benefits=data["Income from Unemployment Benefits"].values,
        income=data["Income"].values,
        expected_income=data["Income"].values,
        income_histogram=get_histogram(data["Income"].values, scale),
        #
        labour_inputs=data["Labour Inputs"].values,
        reservation_wages=data["Employee Income"].values + data["Income from Unemployment Benefits"].values,

        # Personal Income Tax
        employee_income_gross=np.zeros(len(data)),
        taxable_income=np.zeros(len(data)),
        taxable_income_employment=np.zeros(len(data)),
        taxable_income_unemployment=np.zeros(len(data)),
        taxable_income_rental=np.zeros(len(data)),
        # taxable_income_investment=np.zeros(len(data)),         # TODO: implement once other income streams are working
        personal_income_tax_owed=np.zeros(len(data)),
        personal_income_tax_paid=np.zeros(len(data)),
        personal_income_tax_paid_employment=np.zeros(len(data)),
        personal_income_tax_paid_unemployment=np.zeros(len(data)),
        personal_income_tax_adjustment=np.zeros(len(data)),
    )

def create_individuals_states_timeseries(
    states: dict[str, float | np.ndarray | list[np.ndarray] | Any],
    
) -> TimeSeries:
    """Create time series for tracking individual state variables.

    Initializes tracking for state variables

    Args:
        states (dict[str, float | np.ndarray | list[np.ndarray] | Any]): Agent states

    Returns:
        TimeSeries: Initialized time series for individual variables
    """
    return TimeSeries(
        education = convert_enum_to_numpy(states["Education"]),             # necessary for states in individual_properties.py
        activity_status = convert_enum_to_numpy(states["Activity Status"]), # necessary for states in individual_properties.py
        employment_industry = np.nan_to_num(states["Employment Industry"], nan=-1),
        corr_hh = states["Corresponding Household ID"],
    )

def convert_enum_to_numpy(temp_enum: list[Enum])-> np.ndarray:
    """Convert Enum class values to int

    Args:
        temp_enum (list[Enum]): Emum list

    Returns:
        temp_np: numpy array of Enum values
    """

    temp_np = np.empty(len(temp_enum), np.int8)
    for i in range(len(temp_enum)):
        temp_np[i] = int(temp_enum[i].value)

    return temp_np
