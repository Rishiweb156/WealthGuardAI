"""Budget Alert System"""

import pandas as pd
from datetime import datetime
from typing import List, Dict

class BudgetAlertSystem:
    """Monitor spending against budgets and generate alerts."""
    
    def __init__(self, budgets: Dict[str, float]):
        """
        Initialize budget system.
        
        Args:
            budgets: Dict mapping category to monthly budget
                    e.g., {"Groceries": 5000, "Shopping": 10000}
        """
        self.budgets = budgets
    
    def check_alerts(self, df: pd.DataFrame) -> List[Dict]:
        """
        Check current spending against budgets.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Filter to current month
        df['parsed_date'] = pd.to_datetime(df['parsed_date'])
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        df_current = df[
            (df['parsed_date'].dt.month == current_month) &
            (df['parsed_date'].dt.year == current_year)
        ]
        
        # Check each category
        for category, budget_limit in self.budgets.items():
            spent = df_current[df_current['category'] == category]['Withdrawal (INR)'].sum()
            
            percentage = (spent / budget_limit * 100) if budget_limit > 0 else 0
            remaining = budget_limit - spent
            
            # Generate alerts based on thresholds
            if spent > budget_limit:
                alerts.append({
                    'category': category,
                    'severity': 'critical',
                    'spent': spent,
                    'budget': budget_limit,
                    'overspend': spent - budget_limit,
                    'percentage': percentage,
                    'message': f"🚨 OVER BUDGET: {category} - Spent ₹{spent:,.2f} (₹{spent-budget_limit:,.2f} over)"
                })
            elif percentage >= 90:
                alerts.append({
                    'category': category,
                    'severity': 'high',
                    'spent': spent,
                    'budget': budget_limit,
                    'remaining': remaining,
                    'percentage': percentage,
                    'message': f"⚠️ WARNING: {category} - 90% of budget used (₹{remaining:,.2f} left)"
                })
            elif percentage >= 75:
                alerts.append({
                    'category': category,
                    'severity': 'medium',
                    'spent': spent,
                    'budget': budget_limit,
                    'remaining': remaining,
                    'percentage': percentage,
                    'message': f"💡 NOTICE: {category} - 75% of budget used (₹{remaining:,.2f} left)"
                })
        
        return sorted(alerts, key=lambda x: x['percentage'], reverse=True)
    
    def get_budget_summary(self, df: pd.DataFrame) -> Dict:
        """Get overall budget status summary."""
        alerts = self.check_alerts(df)
        
        total_budget = sum(self.budgets.values())
        total_spent = sum(alert['spent'] for alert in alerts)
        
        return {
            'total_budget': total_budget,
            'total_spent': total_spent,
            'remaining': total_budget - total_spent,
            'percentage_used': (total_spent / total_budget * 100) if total_budget > 0 else 0,
            'alerts_count': len(alerts),
            'critical_alerts': len([a for a in alerts if a['severity'] == 'critical']),
            'high_alerts': len([a for a in alerts if a['severity'] == 'high']),
            'status': 'good' if total_spent < total_budget * 0.9 else 'warning'
        }