"""
Pydantic models for the /predict API endpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    """
    Features for one credit card holder.

    Field names and value ranges mirror the UCI Taiwan Credit Card Default
    dataset (ID column excluded).

    Payment status codes (PAY_0 … PAY_6):
        -2 = no consumption
        -1 = paid in full
         0 = revolving credit used
         1 = one month late
         … up to 9 = nine or more months late
    """

    limit_bal: float = Field(
        ..., gt=0, description="Credit limit (NT dollar)", example=50_000
    )
    age: int = Field(..., ge=18, le=100, description="Age in years", example=35)
    education: int = Field(
        ...,
        ge=1,
        le=4,
        description="1=grad school, 2=university, 3=high school, 4=other",
        example=2,
    )
    marriage: int = Field(
        ..., ge=1, le=3, description="1=married, 2=single, 3=other", example=2
    )

    # Most-recent-month payment status (PAY_0) down to 6 months ago (PAY_6)
    pay_0: int = Field(..., ge=-2, le=9, description="Payment status: most recent month", example=0)
    pay_2: int = Field(..., ge=-2, le=9, description="Payment status: 2 months ago", example=0)
    pay_3: int = Field(..., ge=-2, le=9, description="Payment status: 3 months ago", example=0)
    pay_4: int = Field(..., ge=-2, le=9, description="Payment status: 4 months ago", example=0)
    pay_5: int = Field(..., ge=-2, le=9, description="Payment status: 5 months ago", example=0)
    pay_6: int = Field(..., ge=-2, le=9, description="Payment status: 6 months ago", example=-1)

    # Bill amounts (NT dollar), most recent first
    bill_amt1: float = Field(..., ge=0, description="Bill amount: most recent month", example=5000)
    bill_amt2: float = Field(..., ge=0, description="Bill amount: 2 months ago", example=4800)
    bill_amt3: float = Field(..., ge=0, description="Bill amount: 3 months ago", example=4600)
    bill_amt4: float = Field(..., ge=0, description="Bill amount: 4 months ago", example=4400)
    bill_amt5: float = Field(..., ge=0, description="Bill amount: 5 months ago", example=4200)
    bill_amt6: float = Field(..., ge=0, description="Bill amount: 6 months ago", example=4000)

    # Payment amounts (NT dollar), most recent first
    pay_amt1: float = Field(..., ge=0, description="Payment made: most recent month", example=2000)
    pay_amt2: float = Field(..., ge=0, description="Payment made: 2 months ago", example=2000)
    pay_amt3: float = Field(..., ge=0, description="Payment made: 3 months ago", example=2000)
    pay_amt4: float = Field(..., ge=0, description="Payment made: 4 months ago", example=2000)
    pay_amt5: float = Field(..., ge=0, description="Payment made: 5 months ago", example=2000)
    pay_amt6: float = Field(..., ge=0, description="Payment made: 6 months ago", example=2000)

    def to_feature_dict(self) -> dict[str, float | int]:
        """
        Return a dict with the uppercase column names expected by the
        preprocessing pipeline (matching data/credit_default.csv headers).
        """
        return {
            "LIMIT_BAL": self.limit_bal,
            "AGE": self.age,
            "EDUCATION": self.education,
            "MARRIAGE": self.marriage,
            "PAY_0": self.pay_0,
            "PAY_2": self.pay_2,
            "PAY_3": self.pay_3,
            "PAY_4": self.pay_4,
            "PAY_5": self.pay_5,
            "PAY_6": self.pay_6,
            "BILL_AMT1": self.bill_amt1,
            "BILL_AMT2": self.bill_amt2,
            "BILL_AMT3": self.bill_amt3,
            "BILL_AMT4": self.bill_amt4,
            "BILL_AMT5": self.bill_amt5,
            "BILL_AMT6": self.bill_amt6,
            "PAY_AMT1": self.pay_amt1,
            "PAY_AMT2": self.pay_amt2,
            "PAY_AMT3": self.pay_amt3,
            "PAY_AMT4": self.pay_amt4,
            "PAY_AMT5": self.pay_amt5,
            "PAY_AMT6": self.pay_amt6,
        }


class SHAPContribution(BaseModel):
    """SHAP value for a single feature."""

    feature: str
    value: float
    shap_value: float


class PredictionResponse(BaseModel):
    """Response from /predict."""

    prediction: int = Field(..., description="1 = predicted default, 0 = no default")
    default_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model-estimated probability of default"
    )
    shap_contributions: list[SHAPContribution] = Field(
        default_factory=list,
        description="Top feature contributions to this prediction (SHAP values)",
    )
