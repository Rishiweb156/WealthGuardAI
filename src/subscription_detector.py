import pandas as pd
import numpy as np
from datetime import timedelta

class SubscriptionDetector:
    """Detect recurring subscriptions automatically."""
    
    def __init__(self, tolerance_days: int = 3):
        self.tolerance_days = tolerance_days
    
    def detect_subscriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect recurring transactions (subscriptions).
        
        Args:
            df: DataFrame with transaction history
            
        Returns:
            DataFrame with detected subscriptions
        """
        df = df.copy()
        
        # Clean narration for pattern matching
        if 'Narration' in df.columns:
            df['narration_clean'] = df['Narration'].str.replace(r'\d+', '', regex=True)
        else:
             # Fallback if Narration column is missing or named differently
            df['narration_clean'] = df['Description'] if 'Description' in df.columns else ""

        df['parsed_date'] = pd.to_datetime(df['parsed_date'])
        
        # Handle column naming variations
        amount_col = 'Withdrawal (INR)'
        if amount_col not in df.columns and 'amount' in df.columns:
            amount_col = 'amount'
            
        subscriptions = []
        
        # Group by cleaned narration
        for narration in df['narration_clean'].unique():
            if not narration: continue
                
            subset = df[df['narration_clean'] == narration].sort_values('parsed_date')
            
            if len(subset) < 3:  # Need at least 3 occurrences
                continue
            
            # Calculate intervals between transactions
            subset = subset.copy()
            subset['days_between'] = subset['parsed_date'].diff().dt.days
            
            # Check if intervals are consistent
            intervals = subset['days_between'].dropna()
            
            if len(intervals) > 0:
                avg_interval = intervals.mean()
                std_interval = intervals.std()
                
                # Detect monthly subscriptions (~30 days)
                if std_interval < self.tolerance_days and 25 <= avg_interval <= 35:
                    subscriptions.append({
                        'name': subset['Narration'].iloc[0] if 'Narration' in subset.columns else narration,
                        'amount': subset[amount_col].mean(),
                        'frequency': 'Monthly',
                        'interval_days': int(avg_interval),
                        'count': len(subset),
                        'total_spent': subset[amount_col].sum(),
                        'first_charge': subset['parsed_date'].min(),
                        'last_charge': subset['parsed_date'].max(),
                        'next_expected': subset['parsed_date'].max() + timedelta(days=int(avg_interval)),
                        'confidence': 1 - (std_interval / self.tolerance_days)
                    })
                
                # Detect weekly subscriptions (~7 days)
                elif std_interval < self.tolerance_days and 5 <= avg_interval <= 9:
                    subscriptions.append({
                        'name': subset['Narration'].iloc[0] if 'Narration' in subset.columns else narration,
                        'amount': subset[amount_col].mean(),
                        'frequency': 'Weekly',
                        'interval_days': int(avg_interval),
                        'count': len(subset),
                        'total_spent': subset[amount_col].sum(),
                        'first_charge': subset['parsed_date'].min(),
                        'last_charge': subset['parsed_date'].max(),
                        'next_expected': subset['parsed_date'].max() + timedelta(days=int(avg_interval)),
                        'confidence': 1 - (std_interval / self.tolerance_days)
                    })
        
        if not subscriptions:
            return pd.DataFrame()
            
        return pd.DataFrame(subscriptions).sort_values('total_spent', ascending=False)
    
    def calculate_monthly_cost(self, subscriptions_df: pd.DataFrame) -> float:
        """Calculate total monthly subscription cost."""
        if subscriptions_df.empty:
            return 0.0
            
        monthly_cost = 0
        
        for _, sub in subscriptions_df.iterrows():
            if sub['frequency'] == 'Monthly':
                monthly_cost += sub['amount']
            elif sub['frequency'] == 'Weekly':
                monthly_cost += sub['amount'] * 4  # 4 weeks per month
        
        return monthly_cost