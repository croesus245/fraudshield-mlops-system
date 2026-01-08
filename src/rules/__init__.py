"""
Rule Engine - Business logic on top of ML.

Use rules when:
- Blocklists (known bad actors)
- Rate limits (velocity)
- Compliance requirements
"""

from .engine import RuleEngine, Rule, RuleResult, RulePriority

__all__ = ["RuleEngine", "Rule", "RuleResult", "RulePriority"]
