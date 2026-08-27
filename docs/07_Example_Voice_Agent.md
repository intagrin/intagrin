# Example: Voice AI Customer Support

Because IntaGrin is built on top of FastAPI, it can be wired to voice platforms like Twilio, Vapi, or LiveKit via a webhook.

When a customer calls your Twilio number, Twilio sends an HTTP POST with the transcribed text and the caller's phone number as the `session_id` to your IntaGrin `/chat` endpoint.

IntaGrin loads the customer's conversational state from the Postgres memory checkpointer. 

**Non-Blocking Tool Calls:** If the `Billing_Agent` needs to generate a PDF invoice (a slow process), thread-pooling the tool call keeps the FastAPI event loop unblocked — the agent can stream a filler phrase ("Let me check that for you...") back to the caller while the invoice generates in the background. This reduces perceived latency; it doesn't make the invoice generation itself instant.
