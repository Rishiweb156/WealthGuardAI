"""Enhanced FastAPI Backend - WealthGuard AI Platform.

Complete production-ready API with Graph RAG, Predictive Forecasting, and Conversational AI.
ENHANCED with Subscription Detection, Budget Alerts, Merchant Analysis, and Savings Goals.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Existing imports
from src.analyzer import analyze_transactions
from src.categorizer import categorize_transactions
from src.forecaster import AnomalyDetector, FinancialForecaster
from src.graph_engine import FinancialGraph
from src.models import (
    AnalyzerInput,
    CategorizerInput,
    NlpProcessorInput,
    PdfProcessingInput,
    StorytellerInput,
    TimelineInput,
    VisualizerInput,
)
from src.nlp_processor import process_nlp_queries
from src.pdf_parser import process_pdf_statements
from src.storyteller import generate_stories
from src.timeline import build_timeline
from src.visualizer import generate_visualizations

# ✅ NEW: Import enhanced features
try:
    from src.subscription_detector import SubscriptionDetector
    from src.budget_alerts import BudgetAlertSystem
    from src.savings_goals import SavingsGoalTracker, calculate_savings_rate
    from src.merchant_analyzer import MerchantAnalyzer
    from src.anomaly_explainer import ExplainableAnomalyDetector
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="WealthGuard AI Platform",
    description="Predictive Financial Health Analysis with Graph RAG + Enhanced Features",
    version="2.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
CHARTS_DIR = OUTPUT_DIR / "charts"
UI_DIR = Path("ui")

TRANSACTIONS_CSV = OUTPUT_DIR / "all_transactions.csv"
TIMELINE_CSV = OUTPUT_DIR / "timeline.csv"
CATEGORIZED_CSV = OUTPUT_DIR / "categorized.csv"
VIZ_FILE = CHARTS_DIR / "visualization_data.json"
STORIES_FILE = OUTPUT_DIR / "stories.txt"
BUDGETS_FILE = OUTPUT_DIR / "budgets.json"  # ✅ NEW

# Create directories
for directory in [INPUT_DIR, OUTPUT_DIR, ANALYSIS_DIR, CHARTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Mount static files
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")


# --- Request Models ---
class StandardResponse(BaseModel):
    success: bool
    message: str
    result: Optional[dict] = None

class NLPQueryRequest(BaseModel):
    query: str

class FilePathRequest(BaseModel):
    input_csv: str

# ✅ NEW: Enhanced feature request models
class BudgetSetRequest(BaseModel):
    category: str
    amount: float

class BudgetConfig(BaseModel):
    budgets: dict

class SavingsGoalRequest(BaseModel):
    name: str
    target_amount: float
    current_amount: float = 0
    target_date: Optional[str] = None

class GoalUpdateRequest(BaseModel):
    goal_id: int
    amount: float


# Root - Serve Dashboard
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve enhanced HTML dashboard."""
    try:
        html_file = UI_DIR / "index.html"
        if not html_file.exists():
            return HTMLResponse(
                "<h1>Dashboard Not Found</h1><p>Deploy ui/index.html</p>",
                status_code=404
            )
        with open(html_file, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check
@app.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "service": "WealthGuard AI",
        "version": "2.1.0",
        "features": [
            "Graph RAG", 
            "Predictive Forecasting", 
            "Conversational AI",
            "Subscription Detection" if ENHANCED_FEATURES_AVAILABLE else "Basic Features",
            "Budget Alerts" if ENHANCED_FEATURES_AVAILABLE else None,
            "Merchant Analysis" if ENHANCED_FEATURES_AVAILABLE else None
        ],
        "enhanced_features": ENHANCED_FEATURES_AVAILABLE
    }


# ===================
# Core Pipeline APIs
# ===================

@app.post("/parse_pdfs")
async def parse_pdfs(files: List[UploadFile] = File(...)):
    """Parse PDF bank statements."""
    if not files or len(files) > 10:
        raise HTTPException(status_code=400, detail="Provide 1-10 PDF files")
    
    try:
        # Clear input directory
        for f in INPUT_DIR.glob("*.pdf"):
            f.unlink()
        
        # Save PDFs
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Invalid file: {file.filename}")
            
            path = INPUT_DIR / file.filename
            with open(path, "wb") as f:
                f.write(await file.read())
        
        # Process
        result = process_pdf_statements(
            PdfProcessingInput(folder_path=INPUT_DIR, output_csv=TRANSACTIONS_CSV)
        )
        
        return StandardResponse(
            success=True,
            message=f"Processed {len(files)} PDFs",
            result={
                "files": len(files),
                "transactions": sum(len(t) for t in result.transactions),
                "output": str(TRANSACTIONS_CSV)
            }
        )
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build_timeline")
async def build_timeline_api(request: FilePathRequest):
    """Build timeline from transactions."""
    try:
        result = build_timeline(
            TimelineInput(transactions_csv=Path(request.input_csv), output_csv=TIMELINE_CSV)
        )
        return StandardResponse(
            success=True,
            message="Timeline built",
            result={"transactions": len(result.transactions), "output": str(TIMELINE_CSV)}
        )
    except Exception as e:
        logger.error(f"Timeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/categorize_transactions")
async def categorize_api(request: FilePathRequest):
    """Categorize transactions."""
    try:
        result = categorize_transactions(
            CategorizerInput(timeline_csv=Path(request.input_csv), output_csv=CATEGORIZED_CSV)
        )
        return StandardResponse(
            success=True,
            message="Categorized",
            result={"transactions": len(result.transactions), "output": str(CATEGORIZED_CSV)}
        )
    except Exception as e:
        logger.error(f"Categorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_transactions")
async def analyze_api(request: FilePathRequest):
    """Analyze transactions."""
    try:
        result = analyze_transactions(
            AnalyzerInput(input_csv=Path(request.input_csv), output_dir=ANALYSIS_DIR)
        )
        return StandardResponse(
            success=True,
            message="Analysis complete",
            result={
                "patterns": [p.description for p in result.patterns],
                "fees": [f.dict() for f in result.fees],
                "recurring": [r.dict() for r in result.recurring],
                "anomalies": [a.dict() for a in result.anomalies],
                "account_overview": result.account_overview.dict(),
            }
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_visualizations")
async def visualize_api(request: FilePathRequest):
    """Generate visualizations."""
    try:
        result = generate_visualizations(
            VisualizerInput(input_csv=Path(request.input_csv), output_dir=CHARTS_DIR)
        )
        return StandardResponse(
            success=True,
            message="Visualizations ready",
            result={
                "spending_trends": result.spending_trends.dict(),
                "expense_breakdown": result.expense_breakdown.dict(),
                "account_overview": result.account_overview.dict(),
            }
        )
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_stories")
async def stories_api(request: FilePathRequest):
    """Generate financial narrative."""
    try:
        result = generate_stories(
            StorytellerInput(input_csv=Path(request.input_csv), output_file=STORIES_FILE)
        )
        return StandardResponse(
            success=True,
            message="Story generated",
            result={"stories": result.stories}
        )
    except Exception as e:
        logger.error(f"Storyteller error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Advanced AI Features ⭐
# ========================

@app.get("/graph-insights")
async def get_graph_insights():
    """⭐ Graph RAG: Relationship discovery in spending."""
    try:
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No data. Upload PDFs first.")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        graph = FinancialGraph()
        graph.build_graph(df)
        insights = graph.generate_insights()
        
        return {
            "success": True,
            "insights": insights.dict(),
            "graph_data": graph.export_for_visualization()
        }
    except Exception as e:
        logger.error(f"Graph RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast")
async def get_forecast(days: int = 30):
    """⭐ Predictive Forecasting: 30-day balance prediction."""
    try:
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No data. Upload PDFs first.")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        forecaster = FinancialForecaster()
        result = forecaster.forecast_balance(df, days=days)
        
        return {
            "success": True,
            "forecast": result.dict()
        }
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml-anomalies")
async def detect_ml_anomalies():
    """⭐ ML-based anomaly detection."""
    try:
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No data")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        detector = AnomalyDetector()
        anomalies = detector.detect_anomalies(df)
        
        return {
            "success": True,
            "anomalies": anomalies,
            "count": len(anomalies)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversational-query")
async def conversational_query(request: NLPQueryRequest):
    """⭐ Conversational AI: Chat with your finances."""
    try:
        if not CATEGORIZED_CSV.exists():
            return {"success": False, "response": "Please upload bank statements first."}
        
        # Enhanced with subscription detection
        query = request.query.lower()
        df = pd.read_csv(CATEGORIZED_CSV)
        
        # Check if asking about subscriptions
        if ENHANCED_FEATURES_AVAILABLE and ("subscription" in query or "recurring" in query):
            detector = SubscriptionDetector()
            subscriptions = detector.detect_subscriptions(df)
            if len(subscriptions) > 0:
                response = f"Found {len(subscriptions)} recurring payments:\n"
                for _, sub in subscriptions.head(5).iterrows():
                    response += f"- {sub['name']}: ₹{sub['amount']:,.2f}/{sub['frequency']}\n"
                return {"success": True, "response": response}
        
        # Check if asking about merchants
        if ENHANCED_FEATURES_AVAILABLE and "merchant" in query:
            analyzer = MerchantAnalyzer()
            top_merchants = analyzer.get_top_merchants(df, n=5)
            response = "Your top merchants by spending:\n"
            for m in top_merchants:
                response += f"- {m['merchant']}: ₹{m['total_spent']:,.2f} ({m['visit_count']} visits)\n"
            return {"success": True, "response": response}
        
        # Fall back to NLP processor
        result = process_nlp_queries(
            NlpProcessorInput(
                input_csv=CATEGORIZED_CSV,
                query=request.query,
                output_file=OUTPUT_DIR / "nlp_response.txt",
                visualization_file=OUTPUT_DIR / "nlp_viz.json"
            )
        )
        
        return {
            "success": True,
            "response": result.text_response,
            "visualization": result.visualization.dict() if result.visualization else None
        }
    except Exception as e:
        logger.error(f"NLP error: {e}")
        return {"success": False, "response": f"Error: {str(e)}"}


# ========================================
# ✅ NEW: ENHANCED FEATURES ENDPOINTS
# ========================================

@app.get("/subscriptions")
async def get_subscriptions():
    """🆕 Detect recurring subscriptions."""
    if not ENHANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Feature not available. Install enhanced modules.")
    
    try:
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No transaction data found")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        detector = SubscriptionDetector(tolerance_days=3)
        subscriptions = detector.detect_subscriptions(df)
        
        monthly_cost = detector.calculate_monthly_cost(subscriptions)
        
        return {
            "success": True,
            "subscriptions": subscriptions.to_dict('records'),
            "total_monthly_cost": monthly_cost,
            "count": len(subscriptions),
            "potential_savings": monthly_cost * 0.15
        }
    except Exception as e:
        logger.error(f"Subscription detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set-budget")
async def set_budget(request: BudgetSetRequest):
    """🆕 Set budget for a category."""
    try:
        BUDGETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if BUDGETS_FILE.exists():
            with open(BUDGETS_FILE, 'r') as f:
                budgets = json.load(f)
        else:
            budgets = {}
        
        budgets[request.category] = request.amount
        
        with open(BUDGETS_FILE, 'w') as f:
            json.dump(budgets, f, indent=2)
        
        return {
            "success": True,
            "message": f"Budget set for {request.category}",
            "category": request.category,
            "amount": request.amount
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set-budgets")
async def set_multiple_budgets(config: BudgetConfig):
    """🆕 Set multiple budgets at once."""
    try:
        BUDGETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(BUDGETS_FILE, 'w') as f:
            json.dump(config.budgets, f, indent=2)
        
        return {
            "success": True,
            "message": f"Set {len(config.budgets)} budgets",
            "budgets": config.budgets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget-alerts")
async def get_budget_alerts():
    """🆕 Get current budget alerts."""
    if not ENHANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Feature not available")
    
    try:
        if not BUDGETS_FILE.exists():
            return {
                "success": True,
                "alerts": [],
                "message": "No budgets set yet"
            }
        
        with open(BUDGETS_FILE, 'r') as f:
            budgets = json.load(f)
        
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No transaction data")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        alert_system = BudgetAlertSystem(budgets)
        alerts = alert_system.check_alerts(df)
        summary = alert_system.get_budget_summary(df)
        
        return {
            "success": True,
            "alerts": alerts,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Budget alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/savings-goals")
async def create_savings_goal(request: SavingsGoalRequest):
    """🆕 Create a savings goal."""
    if not ENHANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Feature not available")
    
    try:
        tracker = SavingsGoalTracker()
        goal = tracker.add_goal(
            name=request.name,
            target_amount=request.target_amount,
            current_amount=request.current_amount,
            target_date=request.target_date
        )
        
        return {
            "success": True,
            "message": "Savings goal created",
            "goal": goal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/savings-goals")
async def get_savings_goals():
    """🆕 Get all savings goals."""
    if not ENHANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Feature not available")
    
    try:
        tracker = SavingsGoalTracker()
        goals = tracker.get_all_goals()
        summary = tracker.get_summary()
        
        return {
            "success": True,
            "goals": goals,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/merchants")
async def get_merchant_analysis():
    """🆕 Get merchant spending analysis."""
    if not ENHANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Feature not available")
    
    try:
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No transaction data")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        analyzer = MerchantAnalyzer()
        
        top_merchants = analyzer.get_top_merchants(df, n=10)
        clusters = analyzer.identify_spending_clusters(df)
        savings_opportunity = analyzer.calculate_savings_opportunity(df)
        
        return {
            "success": True,
            "top_merchants": top_merchants,
            "spending_clusters": clusters,
            "savings_opportunity": savings_opportunity
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Merchant analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies-explained")
async def get_explained_anomalies():
    """🆕 Get anomalies with explanations."""
    if not ENHANCED_FEATURES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Feature not available")
    
    try:
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No transaction data")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        detector = ExplainableAnomalyDetector(sensitivity=2.5)
        
        anomalies = detector.detect_with_reasons(df)
        summary = detector.get_anomaly_summary(df)
        fraud_patterns = detector.detect_fraud_patterns(df)
        
        return {
            "success": True,
            "anomalies": anomalies.to_dict('records'),
            "summary": summary,
            "fraud_indicators": fraud_patterns
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Data Retrieval APIs
# ====================

@app.get("/transactions")
async def get_transactions():
    """Get all categorized transactions."""
    try:
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=404, detail="No transactions found. Please upload bank statements.")
        
        df = pd.read_csv(CATEGORIZED_CSV)
        return df.to_dict("records")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transaction retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations")
async def get_visualizations():
    """Get visualization data."""
    try:
        if not VIZ_FILE.exists():
            raise HTTPException(status_code=404, detail="No visualizations. Run analysis first.")
        
        with open(VIZ_FILE, encoding="utf-8") as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stories")
async def get_stories():
    """Get financial narrative."""
    try:
        if not STORIES_FILE.exists():
            raise HTTPException(status_code=404, detail="No stories. Generate them first.")
        
        with open(STORIES_FILE, encoding="utf-8") as f:
            return {"stories": f.read().split("\n")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stories retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW: Feature availability check
@app.get("/features")
async def get_available_features():
    """Check which features are available."""
    features = {
        "core": {
            "pdf_parsing": True,
            "categorization": True,
            "analysis": True,
            "visualization": True,
            "storytelling": True
        },
        "advanced": {
            "graph_rag": True,
            "forecasting": True,
            "ml_anomalies": True,
            "conversational_ai": True
        },
        "enhanced": {
            "subscription_detection": ENHANCED_FEATURES_AVAILABLE,
            "budget_alerts": ENHANCED_FEATURES_AVAILABLE,
            "savings_goals": ENHANCED_FEATURES_AVAILABLE,
            "merchant_analysis": ENHANCED_FEATURES_AVAILABLE,
            "explained_anomalies": ENHANCED_FEATURES_AVAILABLE
        }
    }
    
    return {
        "success": True,
        "features": features,
        "version": "2.1.0"
    }


# Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)