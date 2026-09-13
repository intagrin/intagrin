"""condition_functions referenced by routers[].condition (see ai.yaml). Pure and tiny — these run
through runtime/router.py's restricted AST evaluator, so a bad/missing value resolves to False."""


def is_sev1(severity: str | None, blast_radius: int | None = None) -> bool:
    """True for a declared sev1, or a sev2 hitting 3+ services (staffed like a sev1)."""
    sev = (severity or "").strip().lower()
    try:
        radius = int(blast_radius or 0)
    except (TypeError, ValueError):
        radius = 0
    return sev == "sev1" or (sev == "sev2" and radius >= 3)
