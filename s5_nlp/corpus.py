"""A local, offline text corpus for the retrieval topics: this repository's own NOTES.md files.

The retrieval topics evaluate on a FROZEN snapshot (corpus_snapshot.json) so their reported
numbers are reproducible: otherwise every edit to any NOTES.md (including the notes that report
the numbers) would change the corpus. Regenerate deliberately with
    python -m s5_nlp.corpus

Every NOTES.md is split into passages at its "## " headings. The "Interview questions"
sections are held out: each question becomes an evaluation query whose relevant
passages are the other sections of the same topic folder.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).with_name("corpus_snapshot.json")
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


def build_corpus(root: Path = ROOT) -> tuple[list[Passage], list[Query]]:
    """Parse the NOTES.md files as they are right now."""
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


def load_corpus() -> tuple[list[Passage], list[Query]]:
    """The frozen evaluation corpus (falls back to the live notes if no snapshot exists)."""
    if not SNAPSHOT.exists():
        return build_corpus()
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return [Passage(**p) for p in data["passages"]], [Query(**q) for q in data["queries"]]


def write_snapshot() -> None:
    passages, queries = build_corpus()
    data = {"passages": [asdict(p) for p in passages], "queries": [asdict(q) for q in queries]}
    SNAPSHOT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT.name}: {len(passages)} passages, {len(queries)} queries")


def relevant_ids(passages: list[Passage], query: Query) -> set[str]:
    return {p.pid for p in passages if p.topic == query.topic}


if __name__ == "__main__":
    write_snapshot()
