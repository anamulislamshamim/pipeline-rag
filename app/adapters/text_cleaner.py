"""Light text cleanup for arXiv abstracts/titles.

Deliberately minimal: arXiv abstracts are already clean prose. We only fix
the couple of artifacts that consistently show up in the raw feed (line-wrap
newlines turned into spaces, doubled whitespace) rather than attempting
LaTeX rendering, which is out of scope and would risk destroying content
that matters for embeddings.
"""
from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    # arXiv abstracts are hard-wrapped with newlines; collapse to single spaces.
    text = raw.replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def content_hash(title: str, abstract: str) -> str:
    """Stable hash used to detect whether a paper's embedding-relevant content
    changed, so we know whether it needs re-embedding."""
    payload = f"{title.strip()}\u0000{abstract.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
