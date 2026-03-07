"""Budget Guard — tracks extraction spending and enforces cost limits.

Prevents runaway VLM costs by enforcing:
- Per-document cost cap
- Daily spending limit
- Monthly spending limit

Spending history is persisted to a JSON file for audit.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetGuard:
    """Tracks and enforces extraction cost budgets.

    Usage:
        guard = BudgetGuard()
        if guard.check_budget(estimated_cost):
            # proceed with extraction
            guard.add_cost(doc_id, actual_cost, strategy)
        else:
            # skip or use cheaper strategy
    """

    DEFAULT_PER_DOC_LIMIT = 0.50   # USD per document
    DEFAULT_DAILY_LIMIT = 5.00     # USD per day
    DEFAULT_MONTHLY_LIMIT = 20.00  # USD per month

    def __init__(
        self,
        per_doc_limit: Optional[float] = None,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        spending_file: str = ".refinery/spending.json",
    ):
        self.per_doc_limit = per_doc_limit or self.DEFAULT_PER_DOC_LIMIT
        self.daily_limit = daily_limit or self.DEFAULT_DAILY_LIMIT
        self.monthly_limit = monthly_limit or self.DEFAULT_MONTHLY_LIMIT
        self.spending_file = Path(spending_file)
        self._spending = self._load_spending()

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if a proposed cost fits within all budget limits.

        Args:
            estimated_cost: Estimated cost of the next operation in USD.

        Returns:
            True if the cost is within budget.
        """
        # Per-document check
        if estimated_cost > self.per_doc_limit:
            logger.warning(
                "BudgetGuard: Estimated cost $%.4f exceeds per-doc limit $%.2f",
                estimated_cost, self.per_doc_limit,
            )
            return False

        # Daily check
        today_str = date.today().isoformat()
        daily_spent = sum(
            r["cost"] for r in self._spending.get("records", [])
            if r.get("date") == today_str
        )
        if daily_spent + estimated_cost > self.daily_limit:
            logger.warning(
                "BudgetGuard: Daily spend $%.4f + $%.4f would exceed limit $%.2f",
                daily_spent, estimated_cost, self.daily_limit,
            )
            return False

        # Monthly check
        month_str = date.today().strftime("%Y-%m")
        monthly_spent = sum(
            r["cost"] for r in self._spending.get("records", [])
            if r.get("date", "").startswith(month_str)
        )
        if monthly_spent + estimated_cost > self.monthly_limit:
            logger.warning(
                "BudgetGuard: Monthly spend $%.4f + $%.4f would exceed limit $%.2f",
                monthly_spent, estimated_cost, self.monthly_limit,
            )
            return False

        return True

    def add_cost(
        self, doc_id: str, cost: float, strategy: str = "unknown"
    ) -> None:
        """Record a cost entry.

        Args:
            doc_id: Document identifier.
            cost: Cost in USD.
            strategy: Strategy that incurred the cost.
        """
        record = {
            "doc_id": doc_id,
            "cost": round(cost, 6),
            "strategy": strategy,
            "date": date.today().isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._spending.setdefault("records", []).append(record)
        self._save_spending()

        logger.info(
            "BudgetGuard: Recorded $%.4f for %s (%s)",
            cost, doc_id, strategy,
        )

    def get_daily_spend(self) -> float:
        """Total spend for today."""
        today_str = date.today().isoformat()
        return sum(
            r["cost"] for r in self._spending.get("records", [])
            if r.get("date") == today_str
        )

    def get_monthly_spend(self) -> float:
        """Total spend for the current month."""
        month_str = date.today().strftime("%Y-%m")
        return sum(
            r["cost"] for r in self._spending.get("records", [])
            if r.get("date", "").startswith(month_str)
        )

    def _load_spending(self) -> dict:
        """Load spending history from disk."""
        if self.spending_file.exists():
            try:
                with open(self.spending_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"records": []}

    def _save_spending(self) -> None:
        """Persist spending history to disk."""
        self.spending_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.spending_file, "w") as f:
            json.dump(self._spending, f, indent=2)
