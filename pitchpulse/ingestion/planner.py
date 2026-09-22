"""Build a per-match incremental ingestion plan from Bronze fingerprints."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from .fingerprint import fingerprint_match_bundle
from .reader import RawMatchBundle


IngestionStatus = Literal["new", "changed", "unchanged", "removed"]


class MatchBundleReader(Protocol):
    def iter_match_bundles(self) -> Iterator[RawMatchBundle]: ...


@dataclass(frozen=True, slots=True)
class PlannedMatch:
    """One source or checkpoint match classified for the next ingestion run."""

    match_id: int
    status: IngestionStatus
    content_hash: str | None
    checkpoint_hash: str | None


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    """Classification of current source matches and previously seen matches."""

    matches: tuple[PlannedMatch, ...]

    @property
    def new(self) -> tuple[PlannedMatch, ...]:
        return self._with_status("new")

    @property
    def changed(self) -> tuple[PlannedMatch, ...]:
        return self._with_status("changed")

    @property
    def unchanged(self) -> tuple[PlannedMatch, ...]:
        return self._with_status("unchanged")

    @property
    def removed(self) -> tuple[PlannedMatch, ...]:
        return self._with_status("removed")

    @property
    def source_match_ids(self) -> tuple[int, ...]:
        return tuple(
            match.match_id for match in self.matches if match.status != "removed"
        )

    @property
    def ingest_match_ids(self) -> tuple[int, ...]:
        """Return only new and changed matches, preserving source order."""

        return tuple(match.match_id for match in self.to_ingest)

    @property
    def to_ingest(self) -> tuple[PlannedMatch, ...]:
        return tuple(
            match for match in self.matches if match.status in ("new", "changed")
        )

    def matches_for_ingestion(self, *, force: bool = False) -> tuple[PlannedMatch, ...]:
        if force:
            return tuple(match for match in self.matches if match.status != "removed")
        return self.to_ingest

    def _with_status(self, status: IngestionStatus) -> tuple[PlannedMatch, ...]:
        return tuple(match for match in self.matches if match.status == status)


def build_ingestion_plan(
    reader: MatchBundleReader,
    checkpoint_hashes: Mapping[int, str],
) -> IngestionPlan:
    """Fingerprint current matches and classify them against saved checkpoints."""

    planned: list[PlannedMatch] = []
    source_match_ids: set[int] = set()

    for bundle in reader.iter_match_bundles():
        match_id = bundle.match.match_id
        if match_id in source_match_ids:
            raise ValueError(f"duplicate match_id in source selection: {match_id}")
        source_match_ids.add(match_id)

        content_hash = fingerprint_match_bundle(bundle).content_hash
        checkpoint_hash = checkpoint_hashes.get(match_id)
        if checkpoint_hash is None:
            status: IngestionStatus = "new"
        elif checkpoint_hash == content_hash:
            status = "unchanged"
        else:
            status = "changed"
        planned.append(
            PlannedMatch(
                match_id=match_id,
                status=status,
                content_hash=content_hash,
                checkpoint_hash=checkpoint_hash,
            )
        )

    for match_id in sorted(set(checkpoint_hashes) - source_match_ids):
        planned.append(
            PlannedMatch(
                match_id=match_id,
                status="removed",
                content_hash=None,
                checkpoint_hash=checkpoint_hashes[match_id],
            )
        )

    return IngestionPlan(matches=tuple(planned))
