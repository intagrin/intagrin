"""A tiny in-memory mock CRM/billing fixture so this example runs end-to-end with zero external
credentials. Swap this module's lookups for real calls to your CRM/billing API — every tool in
support_tools.py only ever goes through the functions below, never the dict directly."""

from typing import Any

_ACCOUNTS: dict[str, dict[str, Any]] = {
    "cus_1001": {
        "name": "Globex Corporation",
        "tier": "enterprise",
        "plan": "Enterprise Platform",
        "subscribed_services": ["auth-api", "billing-api", "sync-worker"],
        "invoices": {
            "inv_9001": {"amount_due": 128.50, "status": "open"},
            "inv_8890": {"amount_due": 640.00, "status": "open"},
        },
    },
    "cus_1002": {
        "name": "Initech LLC",
        "tier": "standard",
        "plan": "Team",
        "subscribed_services": ["auth-api", "notifications"],
        "invoices": {
            "inv_9002": {"amount_due": 29.00, "status": "open"},
        },
    },
    "cus_1003": {
        "name": "Umbrella Analytics",
        "tier": "enterprise_plus",
        "plan": "Enterprise Platform Plus",
        "subscribed_services": ["auth-api", "billing-api", "search-api", "sync-worker"],
        "invoices": {
            "inv_9003": {"amount_due": 1_240.00, "status": "open"},
        },
    },
}

_SERVICE_STATUS: dict[str, str] = {
    "auth-api": "degraded",
    "billing-api": "operational",
    "sync-worker": "operational",
    "notifications": "operational",
    "search-api": "outage",
}


def get_account(customer_id: str) -> dict[str, Any] | None:
    return _ACCOUNTS.get(customer_id.strip())


def get_invoice(customer_id: str, invoice_id: str) -> dict[str, Any] | None:
    account = get_account(customer_id)
    if not account:
        return None
    return account["invoices"].get(invoice_id.strip())


def mark_invoice_refunded(customer_id: str, invoice_id: str) -> None:
    invoice = get_invoice(customer_id, invoice_id)
    if invoice is not None:
        invoice["status"] = "refunded"


def get_service_status(service_name: str) -> str | None:
    return _SERVICE_STATUS.get(service_name.strip().lower())


def known_service_names() -> list[str]:
    return sorted(_SERVICE_STATUS)
