"""Stable, per-match fingerprints for incremental StatsBomb ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .reader import RawMatchBundle


FINGERPRINT_VERSION = "statsbomb-match-bundle-v1"


@dataclass(frozen=True, slots=True)
class MatchContentFingerprint:
    """Component and aggregate SHA-256 hashes for one raw match bundle."""

    match_id: int
    metadata_hash: str
    events_hash: str
    lineups_hash: str
    three_sixty_hash: str
    content_hash: str


def fingerprint_match_bundle(bundle: RawMatchBundle) -> MatchContentFingerprint:
    """Hash only one match's semantic JSON content, preserving array order.

    Object key order and JSON whitespace do not affect the result. Adding another
    record to the competition-season matches file therefore does not invalidate
    the fingerprints of existing matches.
    """

    metadata_hash = _canonical_sha256(bundle.match.payload)
    events_hash = _canonical_sha256([record.payload for record in bundle.events])
    lineups_hash = _canonical_sha256([record.payload for record in bundle.lineups])
    three_sixty_hash = _canonical_sha256(
        [record.payload for record in bundle.three_sixty]
    )
    content_hash = _canonical_sha256(
        {
            "fingerprint_version": FINGERPRINT_VERSION,
            "match_id": bundle.match.match_id,
            "metadata_hash": metadata_hash,
            "events_hash": events_hash,
            "lineups_hash": lineups_hash,
            "three_sixty_hash": three_sixty_hash,
        }
    )
    return MatchContentFingerprint(
        match_id=bundle.match.match_id,
        metadata_hash=metadata_hash,
        events_hash=events_hash,
        lineups_hash=lineups_hash,
        three_sixty_hash=three_sixty_hash,
        content_hash=content_hash,
    )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
