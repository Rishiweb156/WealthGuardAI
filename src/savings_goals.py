import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SavingsGoalTracker:
    def __init__(self, storage_file: str = "data/output/savings_goals.json"):
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_goals()

    def _load_goals(self):
        if self.storage_file.exists():
            with open(self.storage_file, 'r') as f:
                self.goals = json.load(f)
        else:
            self.goals = []

    def _save_goals(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.goals, f, indent=2)

    def add_goal(self, name: str, target_amount: float, current_amount: float = 0, target_date: str = None) -> Dict:
        goal = {
            "id": len(self.goals) + 1,
            "name": name,
            "target_amount": target_amount,
            "current_amount": current_amount,
            "target_date": target_date,
            "created_at": datetime.now().isoformat(),
            "progress": (current_amount / target_amount) * 100 if target_amount > 0 else 0
        }
        self.goals.append(goal)
        self._save_goals()
        return goal

    def get_all_goals(self) -> List[Dict]:
        return self.goals

    def get_summary(self) -> Dict:
        total_target = sum(g["target_amount"] for g in self.goals)
        total_saved = sum(g["current_amount"] for g in self.goals)
        return {
            "total_goals": len(self.goals),
            "total_saved": total_saved,
            "total_target": total_target,
            "overall_progress": (total_saved / total_target * 100) if total_target > 0 else 0
        }

def calculate_savings_rate(income: float, expenses: float) -> float:
    if income <= 0:
        return 0.0
    return ((income - expenses) / income) * 100