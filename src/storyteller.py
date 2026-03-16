"""Financial Summaries and Narrative Generation.

This module generates a cohesive financial narrative based on analyzed financial data using an LLM.
Key functionalities include:
- Creating an engaging, comprehensive story of spending habits and trends.
- Highlighting key insights and actionable recommendations.
- Formatting narratives for user-friendly presentation.
- Supporting integration with conversational interfaces.
"""
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.models import StorytellerInput, StorytellerOutput
from src.utils import get_llm_config, setup_mlflow
from llm_setup.ollama_manager import get_client, setup_ollama  # ✅ Import the manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_stories(input_model: StorytellerInput) -> StorytellerOutput:
    """Generate a single financial narrative using an LLM.

    Args:
        input_model: StorytellerInput with path to categorized transactions CSV and output file.

    Returns:
        StorytellerOutput with a single story string.

    """
    setup_mlflow()
    logger.info("Generating financial story")

    input_csv = input_model.input_csv
    output_file = input_model.output_file

    with mlflow.start_run(run_name="Storytelling"):
        mlflow.log_param("input_csv", str(input_csv))
        start_time = pd.Timestamp.now()
        try:
            df = pd.read_csv(input_csv)
            logger.info(f"Read CSV: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", input_csv)
            mlflow.log_param("error", f"Input CSV not found: {input_csv}")
            return StorytellerOutput(stories=[])

        if df.empty:
            logger.warning(f"Empty CSV: {input_csv}")
            mlflow.log_param("warning", "Empty CSV")
            return StorytellerOutput(stories=[])

        mlflow.log_metric("transactions_storied", len(df))

        t = pd.Timestamp.now()
        df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
        df["month"] = df["parsed_date"].dt.to_period("M")

        # Aggregate financial data
        total_withdrawals = df["Withdrawal (INR)"].sum()
        total_deposits = df["Deposit (INR)"].sum()
        net_balance = total_deposits - total_withdrawals
        monthly_agg = df.groupby("month").agg({
            "Withdrawal (INR)": "sum",
            "Deposit (INR)": "sum",
            "category": lambda x: x.value_counts().idxmax() if len(x) > 0 else "Unknown",
        }).reset_index()
        monthly_agg["net"] = monthly_agg["Deposit (INR)"] - monthly_agg["Withdrawal (INR)"]
        top_category = df["category"].value_counts().idxmax() if len(df["category"]) > 0 else "Unknown"
        overspending_months = len(monthly_agg[monthly_agg["net"] < 0])
        saving_months = len(monthly_agg[monthly_agg["net"] > 0])

        # Get sample transactions (up to 5 across the period)
        sample_transactions = df[["Narration", "Withdrawal (INR)", "Deposit (INR)", "category", "month"]].head(5).to_dict(orient="records")
        sample_text = "\n".join(
            f"- {t['month']}: {t['Narration']}: ₹{t['Withdrawal (INR)'] or t['Deposit (INR)']} ({t['category']})"
            for t in sample_transactions
        ) if sample_transactions else "No specific transactions available."

        logger.info(f"Aggregate: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # ✅ Initialize LLM using the same manager as categorizer
        llm_config = get_llm_config()
        
        # ✅ First ensure Ollama is set up
        if not setup_ollama(llm_config):
            logger.error("Failed to setup Ollama, generating fallback story")
            mlflow.log_param("llm_error", "Ollama setup failed")
            fallback_story = generate_fallback_story(
                total_withdrawals, total_deposits, net_balance, 
                top_category, overspending_months, saving_months
            )
            # Save fallback story
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(fallback_story)
            mlflow.log_artifact(output_file)
            return StorytellerOutput(stories=[fallback_story])

        # ✅ Use get_client() instead of ollama.Client()
        try:
            client = get_client()
            mlflow.log_param("llm_model", llm_config.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            mlflow.log_param("llm_error", str(e))
            fallback_story = generate_fallback_story(
                total_withdrawals, total_deposits, net_balance,
                top_category, overspending_months, saving_months
            )
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(fallback_story)
            mlflow.log_artifact(output_file)
            return StorytellerOutput(stories=[fallback_story])

        # ✅ Improved prompt with constraints
        t = pd.Timestamp.now()
        prompt = f"""You are a financial advisor. Create a brief, actionable financial summary (150-250 words) for this user's activity from {monthly_agg['month'].min()!s} to {monthly_agg['month'].max()!s}.

Financial Data:
- Total Spending: ₹{total_withdrawals:.2f}
- Total Income: ₹{total_deposits:.2f}
- Net Balance: ₹{net_balance:.2f} ({'savings' if net_balance > 0 else 'deficit'})
- Top Category: {top_category}
- Months with Deficit: {overspending_months}
- Months with Savings: {saving_months}

Sample Transactions:
{sample_text}

Write a friendly 2-3 paragraph summary that:
1. Opens with overall financial health
2. Identifies ONE key pattern or concern
3. Provides ONE specific actionable recommendation

Keep it conversational and motivating. Do not use bullet points."""

        try:
            # ✅ Use chat() instead of generate() for better performance
            response = client.chat(
                model=llm_config.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.7,
                    "num_predict": 300,  # ✅ Limit response length
                    "top_k": 40,
                    "top_p": 0.9,
                }
            )
            story = response["message"]["content"].strip()
            
            if not story:
                raise ValueError("Empty LLM response")
                
            logger.info(f"Story generation: {(pd.Timestamp.now() - t).total_seconds():.3f}s")
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            mlflow.log_param("llm_error", str(e))
            story = generate_fallback_story(
                total_withdrawals, total_deposits, net_balance,
                top_category, overspending_months, saving_months
            )

        # Save story
        t = pd.Timestamp.now()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(story)
        mlflow.log_artifact(output_file)
        logger.info(f"Save: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        logger.info(f"Total: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        return StorytellerOutput(stories=[story])


def generate_fallback_story(total_withdrawals, total_deposits, net_balance, 
                            top_category, overspending_months, saving_months) -> str:
    """Generate a template-based story when LLM is unavailable."""
    
    health_status = "doing well" if net_balance > 0 else "showing a deficit"
    action = "Keep up the good work" if net_balance > 0 else "Consider reviewing your spending"
    
    story = f"""Financial Summary

Your financial health is {health_status} with total income of ₹{total_deposits:,.2f} and expenses of ₹{total_withdrawals:,.2f}, resulting in a net balance of ₹{net_balance:,.2f}.

Your spending is primarily concentrated in {top_category}. Over this period, you had {saving_months} months with positive savings and {overspending_months} months where expenses exceeded income.

Recommendation: {action} by setting monthly budgets for your top spending categories. {"Focus on maintaining your savings discipline." if net_balance > 0 else "Look for opportunities to reduce discretionary spending and build an emergency fund."}
"""
    return story.strip()


if __name__ == "__main__":
    input_model = StorytellerInput(
        input_csv=Path("data/output/categorized.csv"),
        output_file=Path("data/output/stories.txt"),
    )
    output = generate_stories(input_model)
    print(output.stories)