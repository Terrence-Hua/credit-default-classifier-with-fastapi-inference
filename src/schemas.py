"""
Pydantic request and response schemas for the FastAPI inference service.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CreditFeatures(BaseModel):
    """Input features for a single credit-default prediction request."""

    # Credit limit and demographics
    LIMIT_BAL: float = Field(..., description="Credit limit (NT dollar)")
    AGE: int = Field(..., ge=18, le=100, description="Age in years")

    # Bill amounts (last 6 months)
    BILL_AMT1: float = Field(..., description="Bill amount month 1")
    BILL_AMT2: float = Field(..., description="Bill amount month 2")
    BILL_AMT3: float = Field(..., description="Bill amount month 3")
    BILL_AMT4: float = Field(..., description="Bill amount month 4")
    BILL_AMT5: float = Field(..., description="Bill amount month 5")
    BILL_AMT6: float = Field(..., description="Bill amount month 6")

    # Payment amounts (last 6 months)
    PAY_AMT1: float = Field(..., ge=0, description="Payment amount month 1")
    PAY_AMT2: float = Field(..., ge=0, description="Payment amount month 2")
    PAY_AMT3: float = Field(..., ge=0, description="Payment amount month 3")
    PAY_AMT4: float = Field(..., ge=0, description="Payment amount month 4")
    PAY_AMT5: float = Field(..., ge=0, description="Payment amount month 5")
    PAY_AMT6: float = Field(..., ge=0, description="Payment amount month 6")

    # Repayment status (-2=no use, -1=paid duly, 0=revolving, 1-9=months late)
    PAY_0: int = Field(..., ge=-2, le=9, description="Repayment status month 1")
    PAY_2: int = Field(..., ge=-2, le=9, description="Repayment status month 2")
    PAY_3: int = Field(..., ge=-2, le=9, description="Repayment status month 3")
    PAY_4: int = Field(..., ge=-2, le=9, description="Repayment status month 4")
    PAY_5: int = Field(..., ge=-2, le=9, description="Repayment status month 5")
    PAY_6: int = Field(..., ge=-2, le=9, description="Repayment status month 6")

    # Categorical
    EDUCATION: int = Field(
        ..., ge=0, le=6,
        description="Education level: 1=grad, 2=university, 3=high school, 0/4/5/6=other"
    )
    MARRIAGE: int = Field(
        ..., ge=0, le=3,
        description="Marital status: 1=married, 2=single, 3=other, 0=unknown"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 50000,
                "AGE": 35,
                "BILL_AMT1": 15000,
                "BILL_AMT2": 14000,
                "BILL_AMT3": 13000,
                "BILL_AMT4": 12000,
                "BILL_AMT5": 11000,
                "BILL_AMT6": 10000,
                "PAY_AMT1": 2000,
                "PAY_AMT2": 1500,
                "PAY_AMT3": 1500,
                "PAY_AMT4": 1500,
                "PAY_AMT5": 1500,
                "PAY_AMT6": 1500,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "EDUCATION": 2,
                "MARRIAGE": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    default_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of credit default"
    )
    prediction: int = Field(
        ..., ge=0, le=1,
        description="Binary prediction (1=default, 0=no default)"
    )
    model: str = Field(..., description="Model name used for inference")
    shap_values: Optional[dict[str, float]] = Field(
        None, description="SHAP feature contributions (top features only)"
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response from the /model-info endpoint."""

    model_name: str
    holdout_roc_auc: float
    cv_roc_auc: float
    features: list[str]
    train_size: int
    test_size: int
