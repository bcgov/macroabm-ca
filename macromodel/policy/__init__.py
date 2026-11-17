"""
Personal Income Tax module

This module implements policies for personal income taxation.
"""

from macromodel.policy.personal_income_tax import (
    PersonalIncomeTaxCredit,
    PersonalIncomeTax, 
    PersonalIncomeTaxProtocol,
    ProgressiveRate, 
    FlatRate,
)

__all__ = [
    "PersonalIncomeTaxCredit",
    "PersonalIncomeTax", 
    "PersonalIncomeTaxProtocol",
    "ProgressiveRate", 
    "FlatRate",
    ]
