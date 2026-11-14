# Final human-validated real-world dataset

This directory is generated from the two human annotation exports listed in
`manifest.json`. The source annotations are not modified.

## Primary benchmark files

- `events.csv`: 50 validated disaster events.
- `articles.jsonl`: 150 benchmark articles, exactly
  3 distinct publisher domains per event.
- `all_accepted_articles.jsonl`: all 182
  human-accepted articles, including the unused buffer articles.
- `manifest.json`: selection rules, distributions, excluded events, file
  provenance, and SHA-256 checksums of the annotation inputs.

Use the `content` field as the final cleaned article text. Original article text
is intentionally excluded from the public dataset; only its hash and derived
editing statistics are retained. The private annotation exports remain the audit
source. Every row in `articles.jsonl` has human decision `accept`, all five
review criteria set to `yes`, non-empty content, and a source domain distinct
within its event.
