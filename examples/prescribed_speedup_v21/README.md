# Skew-symmetric transformer and body-fitted baseline records

These files hold every timed run behind the body-fitted tables of the
prescribed-flow CHT study after the transformer transport changed to its
skew-symmetric form, together with the v20 runs that the tables keep. The
[study page](../../docs/prescribed-speedup-v21.md) explains the comparisons,
definitions and reproduction commands.

| File | Content |
| --- | --- |
| [`evidence.tar.gz`](evidence.tar.gz) | One digest per run with the SHA-256 of its source record, the study protocols, the job lists, the stability certificates, the independent verification, the transfer replays and the time-refinement summaries |
| [`populations.md`](populations.md) | Every comparison with converged and attempted repetitions per method and the resulting speedup |
| [`rows.json`](rows.json) | The table rows, their comparison keys and the advective-to-skew twins |
| [`summary.json`](summary.json) | Per-method statistics of every comparison |
| [`manifest.json`](manifest.json) | Checksums of these files and the source commits |
