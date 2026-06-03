from typing import Literal, Optional

from pydantic import BaseModel, Field


class SocialBenefits(BaseModel):
    name: Literal["ConstantSocialBenefitsSetter", "DefaultSocialBenefitsSetter", "GrowthSocialBenefitsSetter"] = (
        "GrowthSocialBenefitsSetter"
    )
    path_name: str = "social_benefits"
    parameters: dict = {}


class SocialHousing(BaseModel):
    name: Literal["DefaultSocialHousing"] = "DefaultSocialHousing"
    path_name: str = "social_housing"
    parameters: dict = {"rent_as_fraction_of_unemployment_rate": 0.25}


class CentralGovernmentFunctions(BaseModel):
    social_benefits: SocialBenefits = SocialBenefits()
    social_housing: SocialHousing = SocialHousing()


class CentralGovernmentConfiguration(BaseModel):
    functions: CentralGovernmentFunctions = CentralGovernmentFunctions()

    # Progressive Personal Income Tax schedule.
    # Each tuple is (bracket_upper_bound, marginal_rate).
    # The last bound should be float("inf") for the top bracket.
    # When None (default), the flat ``Income Tax`` scalar is used for
    # both behavioural decisions and government revenue (backward
    # compatible).  When set, revenue is computed progressively on
    # employee income while wage-setting and after-tax income
    # calculations continue to use the scalar ``Income Tax`` effective
    # rate (which is updated each period to actual / taxable base).
    pit_brackets: Optional[list[tuple[float, float]]] = Field(
        default=None,
        description="Progressive PIT brackets as (upper_bound, marginal_rate). "
        "None means use the flat Income Tax rate.",
    )

    # Basic personal amount for non-refundable tax credit.
    # The credit is basic_deduction × lowest_marginal_rate, subtracted
    # after the progressive calculation.  Inflated alongside brackets
    # when CPI-indexing is active.  None means no credit applied.
    pit_basic_deduction: Optional[float] = Field(
        default=None,
        description="Basic personal amount (non-refundable credit base) for PIT.",
    )

