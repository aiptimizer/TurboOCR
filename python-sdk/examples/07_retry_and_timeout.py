"""RetryPolicy plus client-wide and per-request timeouts."""

from pathlib import Path

from turboocr import Client, RetryPolicy

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

policy = RetryPolicy(
    attempts=5,
    backoff=0.5,
    backoff_cap=4.0,
    jitter=0.1,
    retry_statuses=frozenset({429, 502, 503, 504}),
)

client = Client(retry=policy, timeout=30.0)
response = client.recognize_image(IMAGE, timeout=15.0)
print(
    f"recognized {len(response.results)} items "
    f"(attempts={policy.attempts}, backoff={policy.backoff}s, "
    f"cap={policy.backoff_cap}s, "
    f"retry_statuses={sorted(policy.retry_statuses)})"
)

# Output:
# recognized 71 items (attempts=5, backoff=0.5s, cap=4.0s, retry_statuses=[429, 502, 503, 504])
