"""Transaction Categorization Using LLM - OPTIMIZED VERSION
This module categorizes transactions into predefined categories using a
Language Model (LLM) with performance optimizations.
"""

import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.config import LLMConfig
from llm_setup.ollama_manager import query_llm, setup_ollama, get_client
from src.models import CategorizedTransaction, CategorizerInput, CategorizerOutput
from src.utils import (
    ensure_no_active_run,
    get_llm_config,
    sanitize_metric_name,
    setup_mlflow,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_config() -> dict:
    """Load configuration from config file."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {
            "transaction_categories": {
                "income": ["Income (Other)"],
                "essential_expenses": ["Expense (Other)"],
            },
        }


def get_all_categories(config: dict) -> list[str]:
    """Get a flattened list of all transaction categories from config."""
    categories = []
    cat_groups = config.get("transaction_categories", {})
    for group in cat_groups.values():
        categories.extend(group)
    return categories


def get_category_keywords(config: dict) -> dict[str, list[str]]:
    """Get category keywords mapping from config."""
    return config.get("category_keywords", {})


def apply_rules(row: pd.Series, config: dict) -> str:
    """Apply rule-based categorization to a single transaction."""
    narration = str(row["Narration"]).upper()
    withdrawal = row["Withdrawal (INR)"] if not pd.isna(row["Withdrawal (INR)"]) else 0
    deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0

    category_keywords = get_category_keywords(config)

    for category, keywords in category_keywords.items():
        if (category.startswith("Income") and deposit > 0) or \
           ((category.startswith("Expense") or category == "Savings/Investment") and withdrawal > 0):
            if any(kw in narration for kw in keywords):
                return category

    if deposit > 0:
        return "Income (Other)"
    if withdrawal > 0:
        return "Expense (Other)"

    return ""


# ✅ OPTIMIZATION 1: Simplified, shorter prompt
def create_optimized_prompt(desc: str, amount: float, valid_categories: list[str]) -> str:
    """Create a concise prompt for faster LLM response."""
    direction = "Income" if amount > 0 else "Expense"
    
    # ✅ Much shorter prompt = faster response
    prompt = f"""Categorize this transaction. Reply with ONLY the category name, nothing else.

Transaction: {desc}
Amount: ₹{amount:.2f} INR ({direction})

Valid Categories:
{', '.join(valid_categories)}

Category:"""
    
    return prompt


# ✅ OPTIMIZATION 2: Batch processing with threading
def categorize_single_transaction(idx: int, row: pd.Series, llm_config: LLMConfig, 
                                  valid_categories: list[str]) -> tuple[int, str]:
    """Categorize a single transaction using LLM."""
    desc = row["Narration"]
    withdrawal = row["Withdrawal (INR)"] if not pd.isna(row["Withdrawal (INR)"]) else 0
    deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0
    amount = deposit - withdrawal

    prompt = create_optimized_prompt(desc, amount, valid_categories)

    try:
        client = get_client()
        
        # ✅ OPTIMIZATION 3: Constrained generation for speed
        response = client.chat(
            model=llm_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,      # ✅ Lower = faster, more consistent
                "num_predict": 20,       # ✅ Limit response length
                "top_k": 10,             # ✅ Smaller sampling space
                "top_p": 0.5,            # ✅ More focused
                "num_ctx": 512,          # ✅ Smaller context window
            }
        )
        
        llm_response = response["message"]["content"].strip()
        
        # Match to valid category
        matched_category = None
        for valid_cat in valid_categories:
            if valid_cat.lower() in llm_response.lower():
                matched_category = valid_cat
                break

        if matched_category:
            logger.info(
                "Transaction '%s' (%.2f INR) → '%s'",
                desc[:30] + "..." if len(desc) > 30 else desc,
                amount,
                matched_category,
            )
            return (idx, matched_category)
        else:
            default_category = "Income (Other)" if amount > 0 else "Expense (Other)"
            logger.warning(
                "Invalid LLM response '%s' for '%s', using '%s'",
                llm_response,
                desc[:20] + "...",
                default_category,
            )
            return (idx, default_category)

    except Exception as e:
        logger.exception("LLM failed for '%s': %s", desc[:20], str(e))
        default_category = "Income (Other)" if amount > 0 else "Expense (Other)"
        return (idx, default_category)


# ✅ OPTIMIZATION 4: Parallel processing
def apply_llm_fallback_optimized(transactions_df: pd.DataFrame,
                                 llm_config: LLMConfig,
                                 config: dict) -> None:
    """Use LLM for transaction categorization with parallel processing."""
    valid_categories = get_all_categories(config)
    llm_needed = transactions_df[transactions_df["category"] == ""].index

    if not llm_needed.empty:
        logger.info("Using LLM for %d transactions (parallelized)", len(llm_needed))

        # ✅ Process in parallel with ThreadPoolExecutor
        max_workers = 3  # Process 3 transactions concurrently
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    categorize_single_transaction,
                    idx,
                    transactions_df.loc[idx],
                    llm_config,
                    valid_categories
                ): idx
                for idx in llm_needed
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                try:
                    idx, category = future.result()
                    transactions_df.loc[idx, "category"] = category
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.error(f"Failed to categorize index {idx}: {e}")
                    # Set default
                    row = transactions_df.loc[idx]
                    withdrawal = row["Withdrawal (INR)"] if not pd.isna(row["Withdrawal (INR)"]) else 0
                    deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0
                    amount = deposit - withdrawal
                    transactions_df.loc[idx, "category"] = "Income (Other)" if amount > 0 else "Expense (Other)"


def categorize_transactions(input_model: CategorizerInput) -> CategorizerOutput:
    """Categorize transactions using rules, with optimized LLM as fallback."""
    setup_mlflow()
    llm_config = get_llm_config()
    config = load_config()
    logger.info("Starting transaction categorization")

    timeline_csv = input_model.timeline_csv
    output_csv = input_model.output_csv

    if not setup_ollama(llm_config):
        logger.warning("Ollama setup failed, proceeding with rule-based categorization only")

    ensure_no_active_run()
    with mlflow.start_run(run_name="Transaction_Categorization", nested=True):
        mlflow.log_param("input_csv", timeline_csv)
        mlflow.log_param("llm_model", llm_config.model_name)

        try:
            transactions_df = pd.read_csv(timeline_csv)
            logger.info("Loaded CSV with %d rows", len(transactions_df))
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", timeline_csv)
            mlflow.log_param("error", "Input CSV not found")
            return CategorizerOutput(transactions=[])

        if transactions_df.empty:
            logger.warning("No transactions to categorize")
            mlflow.log_param("warning", "No transactions to categorize")
            return CategorizerOutput(transactions=[])

        valid_categories = get_all_categories(config)
        mlflow.log_param("categories_count", len(valid_categories))
        mlflow.log_param("categories", ", ".join(valid_categories))

        # Rule-based categorization
        start_time = pd.Timestamp.now()
        transactions_df["category"] = transactions_df.apply(
            lambda row: apply_rules(row, config), axis=1,
        )
        rule_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(
            "Rules categorized %d/%d transactions in %.2f seconds",
            len(transactions_df[transactions_df["category"] != ""]),
            len(transactions_df),
            rule_time,
        )

        # ✅ Use optimized LLM fallback
        llm_start = pd.Timestamp.now()
        apply_llm_fallback_optimized(transactions_df, llm_config, config)
        llm_time = (pd.Timestamp.now() - llm_start).total_seconds()
        logger.info(f"LLM categorization completed in {llm_time:.2f}s")

        # Default for uncategorized transactions
        transactions_df.loc[
            transactions_df["category"] == "", "category",
        ] = "Expense (Other)"

        # Count categories
        category_counts = transactions_df["category"].value_counts().to_dict()

        # Log metrics
        for category, count in category_counts.items():
            try:
                sanitized_metric = sanitize_metric_name(f"category_{category}")
                mlflow.log_metric(sanitized_metric, count)
            except Exception:
                logger.exception("Failed to log metric for '%s'", category)

        transactions = [
            CategorizedTransaction(**row.to_dict()) for _, row in transactions_df.iterrows()
        ]

        try:
            transactions_df.to_csv(output_csv, index=False)
            mlflow.log_artifact(output_csv)
            mlflow.log_metric("transactions_categorized", len(transactions_df))
            mlflow.log_metric("rule_processing_time_s", rule_time)
            mlflow.log_metric("llm_processing_time_s", llm_time)
            logger.info("Categorized %d transactions", len(transactions_df))
        except Exception:
            logger.exception("Error saving output")

        return CategorizerOutput(transactions=transactions)


if __name__ == "__main__":
    input_model = CategorizerInput(
        timeline_csv="data/output/timeline.csv",
        output_csv="data/output/categorized.csv",
    )
    transactions_df = categorize_transactions(input_model)