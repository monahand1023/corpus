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
