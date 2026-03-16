"""Explainable Anomaly Detection Module"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

class ExplainableAnomalyDetector:
    """Detect anomalies with human-readable explanations."""
    
    def __init__(self, sensitivity: float = 2.5):
        """
        Initialize detector.
        
        Args:
            sensitivity: Z-score threshold (2.5 = top 1% as anomalies)
        """
        self.sensitivity = sensitivity
    
    def detect_with_reasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies and explain why they're anomalous.
        
        Args:
            df: DataFrame with transactions
            
        Returns:
            DataFrame with anomalies and explanations
        """
        df = df.copy()
        df['parsed_date'] = pd.to_datetime(df['parsed_date'])
        
        # Extract features
        df['hour'] = df['parsed_date'].dt.hour
        df['day_of_week'] = df['parsed_date'].dt.dayofweek
        df['amount'] = df['Withdrawal (INR)'].fillna(0)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate z-scores for each feature
        df['amount_zscore'] = self._calculate_zscore(df['amount'])
        df['hour_zscore'] = self._calculate_zscore(df['hour'])
        df['day_zscore'] = self._calculate_zscore(df['day_of_week'])
        
        # Identify anomalies (any z-score > sensitivity)
        df['is_anomaly'] = (
            (abs(df['amount_zscore']) > self.sensitivity) |
            (abs(df['hour_zscore']) > self.sensitivity) |
            (abs(df['day_zscore']) > self.sensitivity)
        )
        
        # Calculate anomaly score (combined z-scores)
        df['anomaly_score'] = (
            abs(df['amount_zscore']) + 
            abs(df['hour_zscore']) + 
            abs(df['day_zscore'])
        ) / 3
        
        # Generate explanations
        df['explanation'] = df.apply(self._explain_anomaly, axis=1)
        
        # Filter to anomalies only
        anomalies = df[df['is_anomaly']].copy()
        
        return anomalies[
            ['parsed_date', 'Narration', 'amount', 'category', 
             'anomaly_score', 'explanation']
        ].sort_values('anomaly_score', ascending=False)
    
    def _calculate_zscore(self, series: pd.Series) -> pd.Series:
        """Calculate z-score for a series."""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series(0, index=series.index)
        
        return (series - mean) / std
    
    def _explain_anomaly(self, row: pd.Series) -> str:
        """Generate human-readable explanation for anomaly."""
        reasons = []
        
        # Check amount anomaly
        if abs(row['amount_zscore']) > self.sensitivity:
            avg_amount = row['amount'] / (abs(row['amount_zscore']) + 1)  # Approximate
            if row['amount_zscore'] > 0:
                reasons.append(
                    f"Unusually HIGH amount (₹{row['amount']:,.2f} vs typical ₹{avg_amount:,.2f})"
                )
            else:
                reasons.append(
                    f"Unusually LOW amount (₹{row['amount']:,.2f} vs typical ₹{avg_amount:,.2f})"
                )
        
        # Check time anomaly
        if abs(row['hour_zscore']) > self.sensitivity:
            typical_hour = 14  # Assume typical shopping hour
            reasons.append(
                f"Unusual time ({row['hour']:02d}:00 vs typical {typical_hour}:00)"
            )
        
        # Check day anomaly
        if abs(row['day_zscore']) > self.sensitivity:
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            reasons.append(
                f"Unusual day ({days[int(row['day_of_week'])]})"
            )
        
        # Check weekend spending
        if row['is_weekend'] == 1 and row['amount'] > 5000:
            reasons.append("Large weekend transaction")
        
        return "; ".join(reasons) if reasons else "Multiple factors"
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics about anomalies.
        
        Args:
            df: DataFrame with transactions
            
        Returns:
            Dict with anomaly statistics
        """
        anomalies = self.detect_with_reasons(df)
        
        if len(anomalies) == 0:
            return {
                'total_anomalies': 0,
                'anomaly_rate': 0,
                'total_anomalous_amount': 0,
                'categories': []
            }
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_rate': (len(anomalies) / len(df)) * 100,
            'total_anomalous_amount': anomalies['amount'].sum(),
            'avg_anomaly_score': anomalies['anomaly_score'].mean(),
            'categories': anomalies['category'].value_counts().to_dict(),
            'top_anomalies': anomalies.head(5).to_dict('records')
        }
    
    def detect_fraud_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect potential fraud patterns.
        
        Args:
            df: DataFrame with transactions
            
        Returns:
            List of potential fraud indicators
        """
        df = df.copy()
        df['parsed_date'] = pd.to_datetime(df['parsed_date'])
        df['amount'] = df['Withdrawal (INR)'].fillna(0)
        
        fraud_indicators = []
        
        # 1. Multiple large transactions in short time
        df_sorted = df.sort_values('parsed_date')
        for i in range(len(df_sorted) - 2):
            window = df_sorted.iloc[i:i+3]
            time_diff = (window['parsed_date'].max() - window['parsed_date'].min()).total_seconds() / 3600
            
            if time_diff < 2 and window['amount'].sum() > 10000:  # 3 transactions in 2 hours > 10k
                fraud_indicators.append({
                    'type': 'rapid_succession',
                    'severity': 'high',
                    'transactions': window[['parsed_date', 'Narration', 'amount']].to_dict('records'),
                    'message': f"3 large transactions within {time_diff:.1f} hours (₹{window['amount'].sum():,.2f})"
                })
        
        # 2. Unusual merchant for category
        df['hour'] = df['parsed_date'].dt.hour
        night_transactions = df[(df['hour'] >= 23) | (df['hour'] <= 5)]
        
        if len(night_transactions) > 0 and night_transactions['amount'].sum() > 5000:
            fraud_indicators.append({
                'type': 'unusual_time',
                'severity': 'medium',
                'transactions': night_transactions[['parsed_date', 'Narration', 'amount']].to_dict('records'),
                'message': f"{len(night_transactions)} transactions between 11 PM - 5 AM (₹{night_transactions['amount'].sum():,.2f})"
            })
        
        # 3. Single very large transaction (> 3 std deviations)
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        
        very_large = df[df['amount'] > mean_amount + (3 * std_amount)]
        
        for _, txn in very_large.iterrows():
            fraud_indicators.append({
                'type': 'unusually_large',
                'severity': 'high',
                'transactions': [txn[['parsed_date', 'Narration', 'amount']].to_dict()],
                'message': f"Very large transaction: ₹{txn['amount']:,.2f} (3x normal)"
            })
        
        return fraud_indicators
    
    def analyze_spending_velocity(self, df: pd.DataFrame, 
                                  window_hours: int = 24) -> Dict:
        """
        Analyze spending velocity (rate of spending).
        
        Args:
            df: DataFrame with transactions
            window_hours: Time window to analyze
            
        Returns:
            Dict with velocity analysis
        """
        df = df.copy()
        df['parsed_date'] = pd.to_datetime(df['parsed_date'])
        df['amount'] = df['Withdrawal (INR)'].fillna(0)
        df = df.sort_values('parsed_date')
        
        # Calculate rolling sum over time window
        df['datetime'] = df['parsed_date']
        df = df.set_index('datetime')
        
        rolling_sum = df['amount'].rolling(
            f'{window_hours}h', 
            closed='left'
        ).sum()
        
        max_velocity = rolling_sum.max()
        max_velocity_time = rolling_sum.idxmax()
        avg_velocity = rolling_sum.mean()
        
        return {
            'window_hours': window_hours,
            'max_velocity': max_velocity,
            'max_velocity_time': str(max_velocity_time),
            'avg_velocity': avg_velocity,
            'velocity_spike': max_velocity / avg_velocity if avg_velocity > 0 else 0,
            'alert': max_velocity > avg_velocity * 3  # 3x normal velocity
        }