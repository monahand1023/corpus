---
title: "Secret scrubbing"
id: scrubbing
---

# Secret scrubbing

Before a chunk is stored or embedded, `corpus` runs `scrub()` over its content to remove obvious secrets and credentials: AWS access keys, GitHub tokens, Stripe live and test keys, Slack tokens, JWTs, OpenAI keys, Voyage keys, Anthropic keys, and PEM private-key blocks all get replaced with a `[REDACTED:...]` marker. A generic `key/secret/token/password = value` assignment pattern is also caught even when it doesn't match one of the named providers.

The threat model is narrow and deliberate: this defends against credential exfiltration if the `.db` file ever leaks, not against hostile content inside chunks. It's tuned to avoid over-redaction, so things that carry real retrieval signal — emails, git SHAs, base64 thumbnails — pass through untouched.

Scrubbing runs once, in the chunker, before the content hash is computed and before anything reaches the embedder — so a leaked secret in your notes never reaches the embedding API or the stored database.
