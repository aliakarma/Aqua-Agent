"""
src/utils/audit_ledger.py
--------------------------
Append-only CSV audit ledger for governance decisions.

Paper Section 3.4:
  "Governance decisions are logged to an append-only audit ledger with
   timestamps and justification codes, supporting regulatory reporting."

Each row records: step, wall-clock time, proposed action, executed action,
override flag, justification code, and per-constraint violation flags.
"""

import csv
import time
from dataclasses import dataclass, astuple, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LedgerEntry:
    step: int
    timestamp: float
    proposed_action: str
    executed_action: str
    overridden: bool
    justification_code: str
    c1_violated: bool      # Consumption cap
    c2_violated: bool      # Fairness
    c3_violated: bool      # Emergency supply


class AuditLedger:
    """
    Append-only CSV audit ledger.

    Usage:
        ledger = AuditLedger("logs/audit_ledger.csv")
        ledger.append(entry)
        df = ledger.to_dataframe()
    """

    FIELDNAMES = [f.name for f in fields(LedgerEntry)]

    def __init__(self, path: str, log_approved: bool = False,
                 log_overrides: bool = True):
        """
        Args:
            path:          File path for the CSV log.
            log_approved:  Whether to log non-overridden (approved) actions.
            log_overrides: Whether to log override events.
        """
        self.path = Path(path)
        self.log_approved = log_approved
        self.log_overrides = log_overrides

        # Counters for summary statistics
        self._total_evaluated = 0
        self._total_overrides = 0
        self._override_by_type: Dict[str, int] = {
            "C1_CAP": 0, "C2_FAIR": 0, "C3_EMRG": 0
        }

        # Create parent dirs and write CSV header if file doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append(self, step: int, proposed_action: str, executed_action: str,
               overridden: bool, justification_code: str,
               c1_violated: bool = False, c2_violated: bool = False,
               c3_violated: bool = False) -> None:
        """Append a governance decision to the ledger."""
        self._total_evaluated += 1
        if overridden:
            self._total_overrides += 1
            if justification_code in self._override_by_type:
                self._override_by_type[justification_code] += 1

        should_log = (overridden and self.log_overrides) or \
                     (not overridden and self.log_approved)
        if not should_log:
            return

        entry = LedgerEntry(
            step=step,
            timestamp=time.time(),
            proposed_action=str(proposed_action),
            executed_action=str(executed_action),
            overridden=overridden,
            justification_code=justification_code,
            c1_violated=c1_violated,
            c2_violated=c2_violated,
            c3_violated=c3_violated,
        )

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {k: v for k, v in zip(self.FIELDNAMES, astuple(entry))}
            )

    def summary(self) -> Dict[str, Any]:
        """Return aggregate override statistics."""
        override_rate = (self._total_overrides / self._total_evaluated
                         if self._total_evaluated > 0 else 0.0)
        return {
            "total_evaluated": self._total_evaluated,
            "total_overrides": self._total_overrides,
            "override_rate": override_rate,
            "overrides_by_type": dict(self._override_by_type),
        }

    def to_dataframe(self):
        """Load ledger into a pandas DataFrame (optional dependency)."""
        try:
            import pandas as pd
            return pd.read_csv(self.path)
        except ImportError:
            raise ImportError("pandas required for to_dataframe(). "
                              "Install with: pip install pandas")

    def query_overrides(self) -> List[LedgerEntry]:
        """Return all override entries as a list of LedgerEntry objects."""
        entries = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["overridden"].lower() == "true":
                    entries.append(LedgerEntry(
                        step=int(row["step"]),
                        timestamp=float(row["timestamp"]),
                        proposed_action=row["proposed_action"],
                        executed_action=row["executed_action"],
                        overridden=True,
                        justification_code=row["justification_code"],
                        c1_violated=row["c1_violated"].lower() == "true",
                        c2_violated=row["c2_violated"].lower() == "true",
                        c3_violated=row["c3_violated"].lower() == "true",
                    ))
        return entries
