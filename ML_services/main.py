from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict
import uvicorn

app = FastAPI(title="Loan Eligibility ML API")

class LoanRequest(BaseModel):
    username: str
    gender: str
    marital_status: str
    dependents: str
    education: str
    age: float
    job_title: str
    annual_salary: float
    collateral_value: float
    savings_balance: float
    employment_type: str
    contract_years: float
    previous_loan: str
    previous_loan_status: str
    previous_loan_amount: float
    total_emi_per_month: float
    loan_purpose: str
    loan_amount: float
    repayment_term_months: float
    additional_income_name: str
    additional_income_amount: float
    num_credit_cards: float
    avg_credit_util_percent: float
    late_payment_history: str
    loan_insurance: str
    credit_score: float
    dti: float
    loan_to_income: float
    rejection_reason: str | None = None
    shap_top3: str | None = None

@app.post("/predict")
def ml_predict(req: LoanRequest):
    output = predict(req.dict())
    return output

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
