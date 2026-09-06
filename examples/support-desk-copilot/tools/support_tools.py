"""Support desk tools. Backed by tools.crm_store's mock fixture data so the whole example runs
with no external credentials — swap crm_store's functions for real CRM/billing API calls when
adapting this to a real deployment."""

import json

from intagrin.errors import AwaitingHumanInput

from . import crm_store

# issue_refund's runtime AwaitingHumanInput threshold (see docs/07_Human_In_The_Loop.md's "Dynamic
# Approval from Inside a Tool") — refunds at or below this amount are low-risk enough to auto-approve.
_AUTO_APPROVE_REFUND_LIMIT = 500.00


def lookup_customer_account(customer_id: str) -> str:
    """Fetches an account's plan tier, subscribed services, and open invoices from the CRM.

    Args:
        customer_id: The account's CRM id (e.g. 'cus_1001').
    """
    account = crm_store.get_account(customer_id)
    if account is None:
        return f"No account found for customer_id '{customer_id}'."
    return json.dumps({"customer_id": customer_id, **account}, indent=2)


def lookup_invoice(customer_id: str, invoice_id: str) -> str:
    """Fetches a single invoice's amount due and status.

    Args:
        customer_id: The account's CRM id (e.g. 'cus_1001').
        invoice_id: The invoice id (e.g. 'inv_9001').
    """
    invoice = crm_store.get_invoice(customer_id, invoice_id)
    if invoice is None:
        return f"No invoice '{invoice_id}' found for customer '{customer_id}'."
    return json.dumps({"invoice_id": invoice_id, **invoice}, indent=2)


def issue_refund(customer_id: str, invoice_id: str, amount: float) -> str:
    """Issues a refund against an invoice. Refunds over the auto-approve limit pause for human
    sign-off instead of gating every call — see AwaitingHumanInput below.

    Args:
        customer_id: The account's CRM id (e.g. 'cus_1001').
        invoice_id: The invoice id being refunded (e.g. 'inv_9001').
        amount: Refund amount in USD.
    """
    invoice = crm_store.get_invoice(customer_id, invoice_id)
    if invoice is None:
        return f"No invoice '{invoice_id}' found for customer '{customer_id}' — nothing to refund."
    if invoice["status"] == "refunded":
        return f"Invoice '{invoice_id}' was already refunded."

    # Decide *before* any side effect: resuming after approval re-invokes this function from the
    # start, so nothing non-idempotent may happen ahead of this check.
    if amount > _AUTO_APPROVE_REFUND_LIMIT:
        raise AwaitingHumanInput(
            prompt=(
                f"Refund of ${amount:.2f} for invoice {invoice_id} (customer {customer_id}) "
                f"exceeds the ${_AUTO_APPROVE_REFUND_LIMIT:.2f} auto-approve limit."
            ),
            context={"customer_id": customer_id, "invoice_id": invoice_id, "amount": amount},
        )

    crm_store.mark_invoice_refunded(customer_id, invoice_id)
    return f"Refunded ${amount:.2f} against invoice '{invoice_id}' for customer '{customer_id}'."


def check_service_status(service_name: str) -> str:
    """Checks the live operational status of an internal platform service.

    Args:
        service_name: The service to check (e.g. 'auth-api', 'billing-api', 'sync-worker',
            'notifications', 'search-api').
    """
    status = crm_store.get_service_status(service_name)
    if status is None:
        known = ", ".join(crm_store.known_service_names())
        return f"Unknown service '{service_name}'. Known services: {known}."
    if status == "operational":
        return f"'{service_name}' is operational — no incidents reported."
    return f"'{service_name}' is currently {status.upper()}. An incident is logged for the platform team."


def auto_close_resolved_ticket(ticket_summary: str) -> str:
    """Closes the current ticket as resolved, once a fix has been confirmed and no untrusted
    diagnostic output is still pending human review (see ai.yaml's available_when gate on this
    tool — the lethal-trifecta guardrail).

    Args:
        ticket_summary: A one-line summary of what was resolved, for the closure record.
    """
    return f"Ticket closed as resolved: {ticket_summary}"
