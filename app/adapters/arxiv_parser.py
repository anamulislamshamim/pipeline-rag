"""Parses the arXiv Atom feed into PaperRecord domain objects.

Deliberately built on stdlib xml.etree.ElementTree rather than feedparser:
one fewer dependency, and full control over arXiv's custom namespace
elements (arxiv:primary_category, arxiv:doi, arxiv:journal_ref), which
feedparser exposes in slightly awkward ad-hoc attributes.

Reference: an arXiv Atom entry looks like:

<entry>
  <id>http://arxiv.org/abs/2401.01234v2</id>
  <updated>2024-01-05T00:00:00Z</updated>
  <published>2024-01-02T00:00:00Z</published>
  <title>Some Paper Title</title>
  <summary>  The abstract, possibly hard-wrapped ...\n</summary>
  <author><name>Jane Doe</name></author>
  <author><name>John Roe</name></author>
  <arxiv:doi>10.1000/xyz</arxiv:doi>                         (optional)
  <arxiv:journal_ref>Some Journal 2024</arxiv:journal_ref>   (optional)
  <link href="http://arxiv.org/abs/2401.01234v2" rel="alternate" type="text/html"/>
  <arxiv:primary_category xmlns:arxiv="..." term="cs.AI"/>
  <category term="cs.AI" scheme="..."/>
  <category term="cs.LG" scheme="..."/>
</entry>
"""
from __future__ import annotations

import re
from datetime import datetime
from xml.etree import ElementTree as ET

from app.adapters.text_cleaner import clean_text, content_hash
from app.domain.models import PaperRecord

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS, "opensearch": OPENSEARCH_NS}

# Matches both new-style ("2401.01234") and old-style ("hep-th/9901001") ids,
# optionally followed by a version suffix like "v2".
_ID_RE = re.compile(r"abs/(?P<arxiv_id>[^v]+?)(?:v(?P<version>\d+))?$")


class ArxivParseError(ValueError):
    """Raised when a feed entry is malformed enough that it can't be parsed."""


def parse_feed(xml_bytes: bytes) -> list[PaperRecord]:
    """Parse a full arXiv API response (an Atom <feed>) into PaperRecords.

    Entries that are individually malformed are skipped rather than aborting
    the whole page, since a single bad record shouldn't lose the rest of the batch.
    """
    root = ET.fromstring(xml_bytes)
    records: list[PaperRecord] = []
    for entry in root.findall("atom:entry", NS):
        try:
            records.append(_parse_entry(entry))
        except ArxivParseError:
            continue
    return records


def total_results(xml_bytes: bytes) -> int:
    root = ET.fromstring(xml_bytes)
    el = root.find("opensearch:totalResults", NS)
    if el is None or el.text is None:
        return 0
    return int(el.text)


def _text(entry: ET.Element, path: str) -> str | None:
    el = entry.find(path, NS)
    if el is None or el.text is None:
        return None
    return el.text.strip()


def _parse_entry(entry: ET.Element) -> PaperRecord:
    raw_id = _text(entry, "atom:id")
    if not raw_id:
        raise ArxivParseError("entry missing <id>")

    match = _ID_RE.search(raw_id)
    if not match:
        raise ArxivParseError(f"could not parse arxiv id from '{raw_id}'")
    arxiv_id = match.group("arxiv_id")
    version = int(match.group("version") or 1)

    title_raw = _text(entry, "atom:title")
    summary_raw = _text(entry, "atom:summary")
    if not title_raw or not summary_raw:
        raise ArxivParseError(f"entry {arxiv_id} missing title/summary")

    title = clean_text(title_raw)
    abstract = clean_text(summary_raw)

    published_raw = _text(entry, "atom:published")
    updated_raw = _text(entry, "atom:updated")
    if not published_raw or not updated_raw:
        raise ArxivParseError(f"entry {arxiv_id} missing published/updated")
    published_at = _parse_datetime(published_raw)
    updated_at = _parse_datetime(updated_raw)

    authors = [
        name.text.strip()
        for author in entry.findall("atom:author", NS)
        if (name := author.find("atom:name", NS)) is not None and name.text
    ]

    categories = [
        cat.get("term")
        for cat in entry.findall("atom:category", NS)
        if cat.get("term")
    ]
    primary_el = entry.find("arxiv:primary_category", NS)
    primary_category = primary_el.get("term") if primary_el is not None else None
    if not primary_category:
        # Fall back to the first listed category if arXiv omits the explicit tag.
        primary_category = categories[0] if categories else "unknown"
    if primary_category not in categories:
        categories.append(primary_category)

    doi = _text(entry, "arxiv:doi")
    journal_ref = _text(entry, "arxiv:journal_ref")

    url = None
    for link in entry.findall("atom:link", NS):
        if link.get("rel") == "alternate":
            url = link.get("href")
            break
    url = url or raw_id

    return PaperRecord(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=authors,
        primary_category=primary_category,
        categories=categories,
        published_at=published_at,
        updated_at=updated_at,
        url=url,
        doi=doi,
        journal_ref=journal_ref,
        content_hash=content_hash(title, abstract),
    )


def _parse_datetime(value: str) -> datetime:
    # arXiv timestamps look like "2024-01-02T00:00:00Z".
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
