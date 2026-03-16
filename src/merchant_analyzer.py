import pandas as pd
from typing import List, Dict

class MerchantAnalyzer:
    def get_top_merchants(self, df: pd.DataFrame, n: int = 5) -> List[Dict]:
        """Group by description/merchant and sum amounts."""
        # Ensure description column exists
        col = 'Description' if 'Description' in df.columns else 'description'
        amount_col = 'Amount' if 'Amount' in df.columns else 'amount'
        
        # Filter for expenses (negative amounts usually, but depends on your CSV format)
        # Assuming Amount is positive for expenses for this analysis, or take abs()
        temp_df = df.copy()
        temp_df[amount_col] = temp_df[amount_col].apply(lambda x: abs(float(x)))
        
        summary = temp_df.groupby(col)[amount_col].agg(['sum', 'count']).reset_index()
        summary.columns = ['merchant', 'total_spent', 'visit_count']
        
        top = summary.sort_values(by='total_spent', ascending=False).head(n)
        return top.to_dict('records')

    def identify_spending_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """Simple rule-based clustering (Low, Medium, High tickets)."""
        amount_col = 'Amount' if 'Amount' in df.columns else 'amount'
        
        clusters = {
            "Small Purchases (< $20)": len(df[abs(df[amount_col]) < 20]),
            "Medium Purchases ($20-$100)": len(df[(abs(df[amount_col]) >= 20) & (abs(df[amount_col]) < 100)]),
            "Large Purchases (> $100)": len(df[abs(df[amount_col]) >= 100])
        }
        return clusters

    def calculate_savings_opportunity(self, df: pd.DataFrame) -> float:
        """Estimate savings if top 10% of discretionary spending was cut."""
        # This is a naive heuristic
        amount_col = 'Amount' if 'Amount' in df.columns else 'amount'
        total_spend = df[amount_col].abs().sum()
        return total_spend * 0.05  # Assume 5% optimization is possible