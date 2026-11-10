"""A local, offline text corpus for the retrieval topics: this repository's own NOTES.md files.

Every NOTES.md is split into passages at its "## " headings. The "Interview questions"
sections are held out: each question becomes an evaluation query whose relevant
passages are the other sections of the same topic folder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION_DIRS = [
    "s1_algorithms",
    "s2_math",
    "s3_classical_ml",
    "s4_deep_learning",
    "s5_nlp",
    "s6_rl",
]


@dataclass(frozen=True)
class Passage:
    pid: str  # e.g. "s3_classical_ml/svm#core-math"
    topic: str  # e.g. "s3_classical_ml/svm"
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    text: str
    answer: str
    topic: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_corpus(root: Path = ROOT) -> tuple[list[Passage], list[Query]]:
    passages, queries = [], []
    for section in SECTION_DIRS:
        for notes in sorted((root / section).glob("*/NOTES.md")):
            topic = f"{section}/{notes.parent.name}"
            title = notes.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
            parts = re.split(r"^## ", notes.read_text(encoding="utf-8"), flags=re.M)[1:]
            for part in parts:
                heading, _, body = part.partition("\n")
                heading = heading.strip()
                if heading.lower().startswith("interview questions"):
                    for q, a in re.findall(
                        r"^\d+\.\s+\*\*(.+?)\*\*\s*(.*?)(?=^\d+\.|\Z)", body, flags=re.M | re.S
                    ):
                        queries.append(Query(q.strip(), " ".join(a.split()), topic))
                    continue
                text = " ".join(body.split())
                if text:
                    passages.append(
                        Passage(f"{topic}#{_slug(heading)}", topic, f"{title} — {heading}", text)
                    )
    return passages, queries


def relevant_ids(passages: list[Passage], query: Query) -> set[str]:
    return {p.pid for p in passages if p.topic == query.topic}
