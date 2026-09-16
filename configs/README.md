# Configuration

- `datasets/`: pinned source and training-corpus manifests.
- `models/`: future model and feature-contract configuration.
- `environments/`: future environment-specific, non-secret configuration.

Secrets belong in environment variables, never in this directory.
