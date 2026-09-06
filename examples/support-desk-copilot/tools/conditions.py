"""condition_functions referenced by routers[].condition (see ai.yaml). Deliberately tiny and
pure — these run through runtime/router.py's restricted AST evaluator, not a real Python call
stack, so no I/O or exceptions worth handling; a bad/missing value should just resolve to False."""


def is_enterprise_tier(customer_tier: str | None) -> bool:
    """True when the account's plan tier qualifies for the priority/enterprise support lane."""
    return (customer_tier or "").strip().lower() in {"enterprise", "enterprise_plus"}
