# Bronze layer

## Purpose

Bronze is the reproducible boundary between StatsBomb Open Data and PitchPulse.
It retains provider JSON without mutation so every downstream table or artifact
can be rebuilt from a pinned source revision.

## Contract

The training-corpus manifest declares:

- a manifest-relative `bronze.data_root`;
- provider format and immutability policy;
- required and optional input families;
- the pinned StatsBomb Git commit;
- competition-season match files, expected counts, and SHA-256 hashes.

At load time, `load_training_corpus_manifest` rejects an unsupported contract,
missing required family, path outside the Bronze root, checksum mismatch,
missing event/lineup input, duplicate match, or incorrect expected summary.
`RawStatsBombValidator` then validates the provider records before any
normalization or PostgreSQL write.

## Promotion boundary

```text
Bronze provider JSON
  -> manifest and file checks
  -> raw cross-record validation
  -> normalize and validate canonical records
  -> PostgreSQL Silver tables
```

Bronze files are read-only inputs. `quarantine.invalid_events` is rejection metadata, not
a corrected Bronze copy. SPADL, action features, labels, and model datasets are
downstream data and must never be written under `data/bronze/`.
