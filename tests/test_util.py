from __future__ import annotations

from corpus.util.dedup import fingerprint, normalize_for_dedup
from corpus.util.scrub import find_secrets, scrub


def test_normalize_strips_urls_dates_whitespace() -> None:
    a = "Hello https://example.com/foo updated 2026-05-01"
    b = "hello  https://different.com/bar  updated 2024-11-15T08:30:00Z"
    assert normalize_for_dedup(a) == normalize_for_dedup(b)


def test_fingerprint_stable() -> None:
    assert fingerprint("hello") == fingerprint("hello")


def test_fingerprint_differs_for_different_content() -> None:
    assert fingerprint("english") != fingerprint("español")


def test_scrub_aws_access_key() -> None:
    text = "set AKIAIOSFODNN7EXAMPLE here"
    out = scrub(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "REDACTED" in out


def test_scrub_github_token() -> None:
    text = "token=ghp_abc123def456ghi789jkl012mno345pq"
    out = scrub(text)
    assert "ghp_" not in out


def test_scrub_preserves_emails_and_git_shas() -> None:
    """Threat model: secrets, not identifying info. Emails + git SHAs survive."""
    text = "Reviewed by dan@example.com in commit 1a2b3c4d5e6f7890abcdef1234567890fedcba98"
    out = scrub(text)
    assert "dan@example.com" in out
    assert "1a2b3c4d5e6f7890abcdef1234567890fedcba98" in out


def test_find_secrets_returns_pattern_name() -> None:
    hits = find_secrets("AKIAIOSFODNN7EXAMPLE and ghp_abc123def456ghi789jkl012mno345pq")
    names = {h[0] for h in hits}
    assert "aws-access-key" in names
    assert "github-token" in names


# --- Redaction coverage, adopted from a consumer repo during consolidation. ---
# `scrub` is the last thing to run before content leaves the machine for an
# embedding or completion API, so a missed pattern is a credential leak. The
# forks that used this engine had 13 cases here and the engine itself had 4;
# these close that gap. Every secret below is a syntactically valid but fake
# credential.


def test_scrub_anthropic_key() -> None:
    text = "key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
    out = scrub(text)
    assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890" not in out
    assert "REDACTED" in out


def test_scrub_voyage_key() -> None:
    out = scrub("voyage_key=pa-abcdefghij1234567890abcdefghij12")
    assert "pa-abcdefghij1234567890abcdefghij12" not in out
    assert "[REDACTED:voyage-key]" in out


def test_scrub_stripe_live_and_test_keys() -> None:
    live = scrub("key=sk_live_abcdefghij1234567890ab")
    test = scrub("key=sk_test_abcdefghij1234567890ab")
    assert "sk_live_abcdefghij1234567890ab" not in live
    assert "[REDACTED:stripe-live-key]" in live
    # The test-mode key is redacted too: it is still a credential, and telling
    # them apart is not this layer's job.
    assert "sk_test_abcdefghij1234567890ab" not in test
    assert "[REDACTED:stripe-test-key]" in test


def test_scrub_slack_token() -> None:
    out = scrub("token=xoxb-abcde12345-fghij67890")
    assert "xoxb-abcde12345-fghij67890" not in out
    assert "[REDACTED:slack-token]" in out


def test_scrub_jwt() -> None:
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiJ1c2VyMTIzNDU2Nzg5MCIsIm5hbWUiOiJKb2huIERvZSJ9"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    out = scrub(f"Authorization: Bearer {jwt}")
    assert jwt not in out
    assert "[REDACTED:jwt]" in out


def test_scrub_private_key_block() -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----\nABCDEFGHIJKLMNOP\n-----END RSA PRIVATE KEY-----"
    out = scrub(text)
    assert "ABCDEFGHIJKLMNOP" not in out
    assert "[REDACTED:private-key]" in out


def test_scrub_generic_assignment_pattern() -> None:
    # Catches a secret that matches no vendor prefix, by the shape of the
    # assignment around it.
    out = scrub('config = { "api_key": "abcdefghij1234567890" }')
    assert "abcdefghij1234567890" not in out


def test_scrub_redacts_several_secrets_in_one_text() -> None:
    text = "aws=AKIAIOSFODNN7EXAMPLE token=ghp_abc123def456ghi789jkl012mno345pq"
    out = scrub(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "ghp_abc123def456ghi789jkl012mno345pq" not in out
    assert "[REDACTED:aws-access-key]" in out


def test_scrub_leaves_surrounding_text_intact() -> None:
    # Redaction replaces the secret, not the sentence around it: the chunk has
    # to stay useful for retrieval after scrubbing.
    out = scrub("set the AKIAIOSFODNN7EXAMPLE here")
    assert out.startswith("set the ")
    assert out.endswith(" here")


def test_scrub_preserves_git_shas() -> None:
    # A 40-hex git SHA is the classic false positive for an over-broad secret
    # regex, and PR descriptions are full of them. They carry retrieval signal
    # and must survive.
    sha = "1a2b3c4d5e6f7890abcdef1234567890fedcba98"
    assert sha in scrub(f"Fix in commit {sha} ; see CI.")


def test_find_secrets_reports_every_matching_pattern() -> None:
    hits = find_secrets("aws=AKIAIOSFODNN7EXAMPLE stripe=sk_live_abcdefghij1234567890ab")
    names = {h[0] for h in hits}
    assert "aws-access-key" in names
    assert "stripe-live" in names
