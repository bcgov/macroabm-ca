from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class TargetCapitalInputsSetter(ABC):
    """Abstract base class for determining firms' capital input targets.

    This class defines strategies for calculating optimal capital input
    demand based on:
    - Production targets and depreciation
    - Current capital stock levels
    - Historical usage patterns
    - Financial constraints

    The capital targeting process considers:
    - Replacement of depreciated capital
    - Capacity expansion needs
    - Credit availability
    - Price expectations

    Attributes:
        target_capital_inputs_fraction (float): Fraction of existing capital
            stock considered available for future production
        credit_gap_fraction (float): How much to reduce capital targets
            when facing credit constraints (between 0 and 1)
    """

    def __init__(
        self,
        target_capital_inputs_fraction: float,
        credit_gap_fraction: float,
        forward_looking_reference_fraction: float = 0.0,
        rolling_reference: bool = False,
    ):
        """Initialize the target capital inputs setter.

        Args:
            target_capital_inputs_fraction (float): Fraction of existing capital
                stock considered available for future production
            credit_gap_fraction (float): How much to reduce capital targets
                when facing credit constraints (between 0 and 1)
            forward_looking_reference_fraction (float): phi in [0, 1]. Weight on
                *desired* (unconstrained target) production, versus realised
                production, when forming the reference capital stock. phi = 0.0
                (default) reproduces the historical backward-looking rule exactly.
            rolling_reference (bool): If True, form the reference stock by scaling
                the firm's *own previous* capital stock by its desired production
                growth rate, K_ref = K_{t-1} * (Q_desired / Q_{t-1}), instead of
                scaling the base-year stock by a base-year production ratio.
                Default False reproduces the base-year rule exactly. When True,
                `forward_looking_reference_fraction` is not used: the rolling rule
                is by construction fully forward-looking in the numerator.
        """
        self.target_capital_inputs_fraction = target_capital_inputs_fraction
        self.credit_gap_fraction = credit_gap_fraction
        self.forward_looking_reference_fraction = forward_looking_reference_fraction
        self.rolling_reference = rolling_reference
        # Safeguard trigger counters (diagnostic only; never affect behaviour).
        self.zero_prev_production_count = 0
        self.zero_prev_capital_count = 0
        self.zero_prev_capital_active_count = 0

    def _reference_capital_stock(
        self,
        unconstrained_target_production: Optional[np.ndarray],
        prev_production: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        """Reference capital stock for the stock-gap term (single source of truth).

        Dispatches to the rolling rule when `rolling_reference` is set and a desired
        production series is available, otherwise to the base-year rule (the
        shipped default, including the phi blend).
        """
        if self.rolling_reference and unconstrained_target_production is not None:
            return self._rolling_reference_capital_stock(
                unconstrained_target_production=unconstrained_target_production,
                prev_production=prev_production,
                prev_capital_inputs_stock=prev_capital_inputs_stock,
                initial_capital_inputs_stock=initial_capital_inputs_stock,
                initial_production=initial_production,
            )
        return self._base_year_reference_capital_stock(
            unconstrained_target_production=unconstrained_target_production,
            prev_production=prev_production,
            initial_capital_inputs_stock=initial_capital_inputs_stock,
            initial_production=initial_production,
        )

    def _base_year_reference_capital_stock(
        self,
        unconstrained_target_production: Optional[np.ndarray],
        prev_production: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        """Base-year reference stock, K_ref = (Q_ref / Q_init) * K_init.

        phi = 0.0 -> Q_ref = prev_production (historical backward-looking rule).
        phi > 0.0 -> Q_ref = (1-phi)*prev_production + phi*desired_production,
                     letting firms hold capital against output they intend to
                     produce rather than only output already realised.
        """
        phi = self.forward_looking_reference_fraction
        if phi > 0.0 and unconstrained_target_production is not None:
            reference_production = (1.0 - phi) * prev_production + phi * np.asarray(
                unconstrained_target_production, dtype=float
            )
        else:
            reference_production = prev_production

        # Guard: where initial_production == 0 the ratio is undefined. Default to
        # 1.0 so the reference stock preserves existing capital (K_ref = K_init)
        # rather than collapsing to zero and shedding the whole stock.
        production_ratio = np.divide(
            reference_production,
            initial_production,
            out=np.ones_like(reference_production, dtype=float),
            where=initial_production != 0.0,
        )
        return production_ratio[:, None] * initial_capital_inputs_stock

    def _rolling_reference_capital_stock(
        self,
        unconstrained_target_production: np.ndarray,
        prev_production: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        """Rolling (recursive) reference capital stock.

            K_ref,t = K_{t-1} * (Q_desired,t / Q_{t-1})

        Unlike the base-year rule, the reference is anchored to the firm's *own*
        previous stock and its desired production *growth rate*. The stock-gap term
        then becomes approximately +lambda * K_{t-1} * g_t, i.e. proportional to
        the stock itself, so it does not vanish at any particular capital level.

        Two safeguards, for the two edge cases this recursion introduces and the
        base-year rule does not have:

        1. Q_{t-1} == 0: the growth ratio is undefined (division by zero). Fall back
           to a ratio of 1.0, giving K_ref = K_{t-1} => gap = 0 => investment falls
           back to pure replacement. This is neutral: it neither sheds the stock nor
           expands it, and simply declines to infer a growth rate from a zero base.

        2. K_{t-1} == 0: the rolling reference degenerates to K_ref = 0, so the gap
           is 0 and the gap term can never rebuild the stock - an absorbing
           zero-capital state (the base-year rule avoids this because its reference
           is keyed to the non-zero base-year stock). Fall back to the base-year
           reference for those entries, which yields gap < 0 and therefore positive
           recovery investment.

        Both safeguards are counted for reporting and never alter behaviour
        elsewhere.
        """
        q_desired = np.asarray(unconstrained_target_production, dtype=float)
        q_prev = np.asarray(prev_production, dtype=float)

        # Safeguard 1: undefined growth ratio on a zero production base.
        zero_prev_production = q_prev == 0.0
        self.zero_prev_production_count += int(np.count_nonzero(zero_prev_production))
        growth_ratio = np.divide(
            q_desired,
            q_prev,
            out=np.ones_like(q_desired, dtype=float),
            where=~zero_prev_production,
        )
        reference_capital_stock = growth_ratio[:, None] * prev_capital_inputs_stock

        # Safeguard 2: absorbing zero-capital state.
        zero_prev_capital = prev_capital_inputs_stock == 0.0
        self.zero_prev_capital_count += int(np.count_nonzero(zero_prev_capital))
        # Of those, the cells where the fallback actually changes behaviour: the
        # firm held this capital type at the base year but has run its stock to
        # zero, so the base-year reference is non-zero and yields a negative gap
        # (recovery investment). Where the base-year stock is itself zero the firm
        # simply never uses this capital type, the fallback reference is also zero,
        # and the safeguard is a no-op. Counted separately so the two are never
        # conflated when reporting.
        self.zero_prev_capital_active_count += int(
            np.count_nonzero(zero_prev_capital & (initial_capital_inputs_stock != 0.0))
        )
        if np.any(zero_prev_capital):
            base_year_ratio = np.divide(
                q_desired,
                initial_production,
                out=np.ones_like(q_desired, dtype=float),
                where=initial_production != 0.0,
            )
            base_year_reference = base_year_ratio[:, None] * initial_capital_inputs_stock
            reference_capital_stock = np.where(zero_prev_capital, base_year_reference, reference_capital_stock)

        return reference_capital_stock

    @abstractmethod
    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
        previous_good_prices: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        extra_taxes: Optional[np.ndarray] = None,
        unconstrained_target_production: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate unconstrained target capital inputs for each firm.

        Determines ideal capital input needs before considering
        financial constraints, based on:
        - Production targets
        - Depreciation rates
        - Current stock levels
        - Historical usage patterns

        Args:
            current_target_production (np.ndarray): Target production levels
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation
                rates for different capital types
            prev_capital_inputs_stock (np.ndarray): Current capital stock
                by type
            initial_capital_inputs_stock (np.ndarray): Initial capital stock
                levels by type
            prev_production (np.ndarray): Previous production levels
            initial_production (np.ndarray): Initial production levels
            previous_good_prices (np.ndarray): Previous period's input prices
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for inputs (not used in Leontief technology)
            extra_taxes (Optional[np.ndarray], optional): Additional taxes on inputs
                that may affect input decisions. Defaults to None.

        Returns:
            np.ndarray: Unconstrained target capital inputs by firm and type
        """
        pass

    @abstractmethod
    def compute_target_capital_inputs(
        self,
        unconstrained_target_capital_inputs: np.ndarray,
        target_long_term_credit: np.ndarray,
        received_long_term_credit: np.ndarray,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> np.ndarray:
        """Calculate financially constrained target capital inputs.

        Adjusts unconstrained targets based on:
        - Credit availability
        - Price expectations
        - Financial constraints

        Args:
            unconstrained_target_capital_inputs (np.ndarray): Ideal capital
                input quantities before financial constraints
            target_long_term_credit (np.ndarray): Desired long-term credit
            received_long_term_credit (np.ndarray): Actually received credit
            previous_good_prices (np.ndarray): Previous period's prices
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Financially constrained target capital inputs
        """
        pass


class FinancialTargetCapitalInputsSetter(TargetCapitalInputsSetter):
    """Implementation of capital input targeting with financial considerations.

    This class implements a strategy that:
    1. Calculates base capital needs from production and depreciation
    2. Adjusts for existing stock levels relative to historical usage
    3. Further adjusts based on credit constraints and price expectations

    The approach ensures that capital targets are:
    - Sufficient for planned production
    - Efficient in stock management
    - Financially feasible
    """

    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
        previous_good_prices: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        extra_taxes: Optional[np.ndarray] = None,
        unconstrained_target_production: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate unconstrained capital input targets with stock adjustment.

        The method:
        1. Calculates base capital needs from production and depreciation
        2. Adjusts for current stock levels relative to historical usage
        3. Ensures non-negative targets

        Args:
            current_target_production (np.ndarray): Target production levels
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation rates
            prev_capital_inputs_stock (np.ndarray): Current stock levels
            initial_capital_inputs_stock (np.ndarray): Reference stock levels
            prev_production (np.ndarray): Current production levels
            initial_production (np.ndarray): Reference production levels
            previous_good_prices (np.ndarray): Previous period's input prices
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for inputs (not used in Leontief technology)
            extra_taxes (Optional[np.ndarray], optional): Additional taxes on inputs
                that may affect input decisions. Defaults to None.

        Returns:
            np.ndarray: Unconstrained target capital inputs by firm and type
        """
        target_capital_inputs = np.multiply(
            current_target_production[:, None],
            capital_inputs_depreciation_matrix,
            out=np.zeros_like(capital_inputs_depreciation_matrix),
        )

        reference_capital_stock = self._reference_capital_stock(
            unconstrained_target_production=unconstrained_target_production,
            prev_production=prev_production,
            prev_capital_inputs_stock=prev_capital_inputs_stock,
            initial_capital_inputs_stock=initial_capital_inputs_stock,
            initial_production=initial_production,
        )

        # Take current stock of capital inputs into accounts
        target_capital_inputs = np.maximum(
            0.0,
            target_capital_inputs
            - self.target_capital_inputs_fraction * (prev_capital_inputs_stock - reference_capital_stock),
        )

        return target_capital_inputs

    def compute_target_capital_inputs(
        self,
        unconstrained_target_capital_inputs: np.ndarray,
        target_long_term_credit: np.ndarray,
        received_long_term_credit: np.ndarray,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> np.ndarray:
        """Calculate financially constrained capital input targets.

        Adjusts unconstrained targets downward based on:
        1. Credit gap (difference between desired and received credit)
        2. Expected price changes
        3. Previous period prices

        Args:
            unconstrained_target_capital_inputs (np.ndarray): Ideal capital
                input quantities before financial constraints
            target_long_term_credit (np.ndarray): Desired long-term credit
            received_long_term_credit (np.ndarray): Actually received credit
            previous_good_prices (np.ndarray): Previous period's prices
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Financially constrained target capital inputs,
                adjusted for credit availability and price expectations
        """
        return np.maximum(
            0.0,
            unconstrained_target_capital_inputs
            - self.credit_gap_fraction
            * (target_long_term_credit - received_long_term_credit)[:, None]
            / ((1 + expected_inflation) * previous_good_prices),
        )


class BundleWeightedTargetCapitalInputsSetter(FinancialTargetCapitalInputsSetter):
    """Implementation of capital input targeting with bundle-based weighting.

    This class extends the financial targeting approach by applying weights to the
    unconstrained targets based on:
    - Input prices
    - Depreciation coefficients
    - Substitution bundles

    The weighting mechanism allows for substitution between capital inputs within the same bundle
    based on relative prices and depreciation rates.
    """

    def __init__(
        self,
        target_capital_inputs_fraction: float,
        credit_gap_fraction: float,
        beta: float = 1.0,
        forward_looking_reference_fraction: float = 0.0,
        rolling_reference: bool = False,
    ) -> None:
        """Initialize the bundle-weighted target capital inputs setter.

        Args:
            target_capital_inputs_fraction (float): Fraction of existing capital
                stock considered available for future production
            credit_gap_fraction (float): How much to reduce capital targets when
                facing credit constraints (between 0 and 1)
            beta (float, optional): Parameter controlling the sensitivity of weights
                to price and depreciation differences. Defaults to 1.0.
            forward_looking_reference_fraction (float): see base class.
            rolling_reference (bool): see base class.
        """
        super().__init__(
            target_capital_inputs_fraction,
            credit_gap_fraction,
            forward_looking_reference_fraction,
            rolling_reference,
        )
        self.beta = beta

    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
        previous_good_prices: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        extra_taxes: Optional[np.ndarray] = None,
        unconstrained_target_production: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate bundle-weighted unconstrained capital input targets.

        The method:
        1. Computes base unconstrained targets using the parent class method
        2. Calculates weights based on prices, depreciation, and bundles
        3. Applies the weights to the unconstrained targets

        Args:
            current_target_production (np.ndarray): Target production levels
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation rates
            prev_capital_inputs_stock (np.ndarray): Current stock levels
            initial_capital_inputs_stock (np.ndarray): Reference stock levels
            prev_production (np.ndarray): Current production levels
            initial_production (np.ndarray): Reference production levels
            previous_good_prices (np.ndarray): Previous period's input prices
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for inputs
            extra_taxes (Optional[np.ndarray], optional): Additional taxes on inputs
                that may affect input decisions. Defaults to None.

        Returns:
            np.ndarray: Bundle-weighted unconstrained target capital inputs
        """

        if extra_taxes is None:
            extra_taxes = np.zeros_like(previous_good_prices)

        # Get base unconstrained targets from parent class
        base_targets = super().compute_unconstrained_target_capital_inputs(
            current_target_production,
            capital_inputs_depreciation_matrix,
            prev_capital_inputs_stock,
            initial_capital_inputs_stock,
            prev_production,
            initial_production,
            previous_good_prices,
            substitution_bundle_matrix,
            extra_taxes,
            unconstrained_target_production,
        )

        # Calculate average price (using nanmean to handle potential NaN values)
        avg_price = np.nanmean(previous_good_prices)

        # Calculate unnormalized weights
        # exp(-beta / avg_price * price[j] / depreciation_matrix[i,j])
        unnormalized_weights = np.exp(-self.beta / avg_price * (previous_good_prices + extra_taxes))
        unnormalized_weights = np.outer(np.ones_like(unnormalized_weights), unnormalized_weights)

        # Create bundle matrix C = M*M.transpose() and replace non-zero coefficients with 1
        n_industries = substitution_bundle_matrix.shape[0]
        bundle_matrix = np.zeros((n_industries, n_industries))

        # Compute C = M*M.transpose()
        bundle_matrix = np.dot(substitution_bundle_matrix, substitution_bundle_matrix.T)

        # Replace non-zero coefficients with 1
        bundle_matrix = np.where(bundle_matrix > 0, 1, 0)

        # Calculate normalization factors for each firm and input
        # sum_l b[i,l] * unnormalized_weights[i,l]
        normalization_factors = np.einsum("jl, il -> ij", bundle_matrix, unnormalized_weights)

        # Calculate normalized weights
        normalized_weights = np.divide(
            unnormalized_weights,
            normalization_factors,
            out=np.ones_like(unnormalized_weights),
            where=normalization_factors != 0,
        )

        normalized_weights *= bundle_matrix.sum(axis=0)

        # Apply weights to base targets
        weighted_targets = base_targets * normalized_weights

        return weighted_targets
