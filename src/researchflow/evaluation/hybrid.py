"""Reciprocal-rank fusion for lexical and embedding rankings."""

from researchflow.tools.offline.interfaces import SearchHit


def rrf_fuse(*rankings: list[SearchHit], constant: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.path] = scores.get(hit.path, 0.0) + 1 / (constant + rank)
            hits.setdefault(hit.path, hit)
    return [
        hits[path] for path in sorted(scores, key=lambda path: (-scores[path], path))
    ]
