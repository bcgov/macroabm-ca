from abc import ABC, abstractmethod
from typing import Protocol

class PersonalIncomeTax(ABC):
    """Abstract base class for personal income tax policy implementations."""

    @abstractmethod
    def calculate_tax(self, income: float) -> float:
        """Calculate tax based on income."""
        pass

    @abstractmethod
    def get_rate(self) -> list[tuple[float, float]]:
        """Return the taxation rate."""
        pass

    @abstractmethod
    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        """Set the taxation rate."""
        pass

class FlatRate(PersonalIncomeTax):
    """Flat rate income tax policy."""
    def __init__(self, brackets: list[tuple[float, float]]):
        """
        brackets: list of tuples (threshold, rate)
        Example: [(float('inf'), 0.15)]
        NOTE: Only threshold must be float('inf') 
        """
        self._validate_brackets(brackets)
        self.brackets = brackets

    def _validate_brackets(self, brackets: list[tuple[float, float]]) -> None:
        if not brackets or len(brackets) != 1:
            raise ValueError("FlatRate must have exactly one bracket.")
        
        threshold, rate = brackets[0]

        if threshold != float('inf'):
            raise ValueError("FlatRate threshold must be float('inf').")
        if not (0 <= rate <= 1):
            raise ValueError("rate must be between 0 and 1.")

    def calculate_tax(self, income: float) -> float:
        if income < 0:
            raise ValueError("Income cannot be negative.")
        _, rate = self.brackets[0]
        return income * rate

    def get_rate(self) -> list[tuple[float, float]]:
        return self.brackets

    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        self._validate_brackets(brackets)        
        self.brackets = brackets

class ProgressiveRate(PersonalIncomeTax):
    """Progressive rate income tax policy with brackets."""
    def __init__(self, brackets: list[tuple[float, float]]):
        """
        brackets: list of tuples (upper threshold, tax rate applied to income below threshold)
        Example: [(50000, 0.1), (100000, 0.2), (float('inf'), 0.3)] 
        NOTE: Final threshold must be float('inf')
        """
        self._validate_brackets(brackets)
        self.brackets = brackets

    def _validate_brackets(self, brackets: list[tuple[float, float]]) -> None:
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
    
    def calculate_tax(self, income: float) -> float:
        if income < 0:
            raise ValueError("Income cannot be negative.")
        tax = 0.0
        prev_threshold = 0.0
        for threshold, rate in self.brackets:
            taxable = min(income, threshold) - prev_threshold
            if taxable > 0:
                tax += taxable * rate
            prev_threshold = threshold
            if income <= threshold:
                break
        return tax

    def get_rate(self) -> list[tuple[float, float]]:
        return self.brackets
    
    def set_rate(self, brackets: list[tuple[float, float]]) -> None:
        self._validate_brackets(brackets)
        self.brackets = brackets

class PersonalIncomeTaxProtocol(Protocol):
    def calculate_tax(self, income: float) -> float: ...
    def get_rate(self) -> list[tuple[float, float]]: ...
    def set_rate(self, brackets: list[tuple[float, float]]) -> None: ...
