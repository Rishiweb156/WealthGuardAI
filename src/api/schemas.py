from pydantic import BaseModel
from typing import Optional, Dict

class StandardResponse(BaseModel):
    success: bool
    message: str
    result: Optional[dict] = None

class NLPQueryRequest(BaseModel):
    query: str

class FilePathRequest(BaseModel):
    input_csv: str

class BudgetSetRequest(BaseModel):
    category: str
    amount: float

class BudgetConfig(BaseModel):
    budgets: Dict[str, float]

class SavingsGoalRequest(BaseModel):
    name: str
    target_amount: float
    current_amount: float = 0
    target_date: Optional[str] = None