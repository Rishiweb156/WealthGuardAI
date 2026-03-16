"""Additional API Routes for Enhanced Features"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

# Import enhancement modules directly from src
from src.subscription_detector import SubscriptionDetector
from src.budget_alerts import BudgetAlertSystem
from src.savings_goals import SavingsGoalTracker, calculate_savings_rate
from src.merchant_analyzer import MerchantAnalyzer
from src.anomaly_explainer import ExplainableAnomalyDetector

router = APIRouter()

# ==================== PYDANTIC MODELS ====================

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

class GoalUpdateRequest(BaseModel):
    goal_id: int
    amount: float

# ==================== SUBSCRIPTION ENDPOINTS ====================

@router.get("/subscriptions")
async def get_subscriptions():
    """Detect and return recurring subscriptions."""
    try:
        df = pd.read_csv("data/output/categorized.csv")
        detector = SubscriptionDetector(tolerance_days=3)
        subscriptions = detector.detect_subscriptions(df)
        
        monthly_cost = detector.calculate_monthly_cost(subscriptions)
        
        return {
            "status": "success",
            "subscriptions": subscriptions.to_dict('records'),
            "total_monthly_cost": monthly_cost,
            "count": len(subscriptions),
            "potential_savings": monthly_cost * 0.15  # Assume 15% can be saved
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No transaction data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subscriptions/{subscription_name}")
async def get_subscription_details(subscription_name: str):
    """Get details for a specific subscription."""
    try:
        df = pd.read_csv("data/output/categorized.csv")
        detector = SubscriptionDetector()
        subscriptions = detector.detect_subscriptions(df)
        
        sub = subscriptions[subscriptions['name'].str.contains(subscription_name, case=False, na=False)]
        
        if len(sub) == 0:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return {
            "status": "success",
            "subscription": sub.iloc[0].to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== BUDGET ENDPOINTS ====================

@router.post("/set-budget")
async def set_budget(request: BudgetSetRequest):
    """Set budget for a specific category."""
    try:
        # Load existing budgets
        budget_file = Path("data/output/budgets.json")
        budget_file.parent.mkdir(parents=True, exist_ok=True)
        
        if budget_file.exists():
            import json
            with open(budget_file, 'r') as f:
                budgets = json.load(f)
        else:
            budgets = {}
        
        # Update budget
        budgets[request.category] = request.amount
        
        # Save budgets
        import json
        with open(budget_file, 'w') as f:
            json.dump(budgets, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Budget set for {request.category}",
            "category": request.category,
            "amount": request.amount
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set-budgets")
async def set_multiple_budgets(config: BudgetConfig):
    """Set multiple budgets at once."""
    try:
        budget_file = Path("data/output/budgets.json")
        budget_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(budget_file, 'w') as f:
            json.dump(config.budgets, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Set {len(config.budgets)} budgets",
            "budgets": config.budgets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/budget-alerts")
async def get_budget_alerts():
    """Get current budget alerts."""
    try:
        # Load budgets
        budget_file = Path("data/output/budgets.json")
        if not budget_file.exists():
            return {
                "status": "success",
                "alerts": [],
                "message": "No budgets set yet"
            }
        
        import json
        with open(budget_file, 'r') as f:
            budgets = json.load(f)
        
        # Load transactions
        df = pd.read_csv("data/output/categorized.csv")
        
        # Check alerts
        alert_system = BudgetAlertSystem(budgets)
        alerts = alert_system.check_alerts(df)
        summary = alert_system.get_budget_summary(df)
        
        return {
            "status": "success",
            "alerts": alerts,
            "summary": summary
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No transaction data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SAVINGS GOALS ENDPOINTS ====================

@router.post("/savings-goals")
async def create_savings_goal(request: SavingsGoalRequest):
    """Create a new savings goal."""
    try:
        tracker = SavingsGoalTracker()
        goal = tracker.add_goal(
            name=request.name,
            target_amount=request.target_amount,
            current_amount=request.current_amount,
            target_date=request.target_date
        )
        
        return {
            "status": "success",
            "message": "Savings goal created",
            "goal": goal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/savings-goals")
async def get_savings_goals():
    """Get all savings goals."""
    try:
        tracker = SavingsGoalTracker()
        goals = tracker.get_all_goals()
        summary = tracker.get_summary()
        
        return {
            "status": "success",
            "goals": goals,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/savings-goals/{goal_id}/progress")
async def update_goal_progress(goal_id: int, request: GoalUpdateRequest):
    """Update progress on a savings goal."""
    try:
        tracker = SavingsGoalTracker()
        goal = tracker.update_progress(request.goal_id, request.amount)
        
        return {
            "status": "success",
            "message": "Goal progress updated",
            "goal": goal
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/savings-goals/{goal_id}/projection")
async def get_goal_projection(goal_id: int):
    """Get projection for when goal will be achieved."""
    try:
        # Calculate average savings rate
        df = pd.read_csv("data/output/categorized.csv")
        avg_savings = calculate_savings_rate(df, months=3)
        
        tracker = SavingsGoalTracker()
        projection = tracker.calculate_projection(goal_id, avg_savings)
        
        return {
            "status": "success",
            "projection": projection
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/savings-goals/{goal_id}")
async def delete_savings_goal(goal_id: int):
    """Delete a savings goal."""
    try:
        tracker = SavingsGoalTracker()
        tracker.delete_goal(goal_id)
        
        return {
            "status": "success",
            "message": f"Goal {goal_id} deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MERCHANT ANALYSIS ENDPOINTS ====================

@router.get("/merchants")
async def get_merchant_analysis():
    """Get merchant spending analysis."""
    try:
        df = pd.read_csv("data/output/categorized.csv")
        analyzer = MerchantAnalyzer()
        
        top_merchants = analyzer.get_top_merchants(df, n=10)
        clusters = analyzer.identify_spending_clusters(df)
        savings_opportunity = analyzer.calculate_savings_opportunity(df)
        
        return {
            "status": "success",
            "top_merchants": top_merchants,
            "spending_clusters": clusters,
            "savings_opportunity": savings_opportunity
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No transaction data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/merchants/{merchant_name}/similar")
async def get_similar_merchants(merchant_name: str):
    """Find similar merchants."""
    try:
        df = pd.read_csv("data/output/categorized.csv")
        analyzer = MerchantAnalyzer()
        
        similar = analyzer.find_similar_merchants(df, merchant_name, n=5)
        
        return {
            "status": "success",
            "merchant": merchant_name,
            "similar_merchants": similar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ANOMALY DETECTION ENDPOINTS ====================

@router.get("/anomalies")
async def get_anomalies():
    """Get detected anomalies with explanations."""
    try:
        df = pd.read_csv("data/output/categorized.csv")
        detector = ExplainableAnomalyDetector(sensitivity=2.5)
        
        anomalies = detector.detect_with_reasons(df)
        summary = detector.get_anomaly_summary(df)
        fraud_patterns = detector.detect_fraud_patterns(df)
        
        return {
            "status": "success",
            "anomalies": anomalies.to_dict('records'),
            "summary": summary,
            "fraud_indicators": fraud_patterns
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No transaction data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/spending-velocity")
async def get_spending_velocity(window_hours: int = 24):
    """Analyze spending velocity."""
    try:
        df = pd.read_csv("data/output/categorized.csv")
        detector = ExplainableAnomalyDetector()
        
        velocity = detector.analyze_spending_velocity(df, window_hours)
        
        return {
            "status": "success",
            "velocity_analysis": velocity
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No transaction data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TRANSACTIONS ENDPOINT (FIX FOR YOUR ISSUE) ====================

@router.get("/transactions")
async def get_transactions(
    category: Optional[str] = None,
    limit: Optional[int] = 100
):
    """Get transaction list with optional filtering."""
    try:
        df = pd.read_csv("data/output/categorized.csv")
        
        # Apply filters
        if category and category != "All":
            df = df[df['category'] == category]
        
        # Limit results
        df = df.head(limit)
        
        # Convert to records
        transactions = df.to_dict('records')
        
        return transactions
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No transaction data found. Please upload bank statements first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))