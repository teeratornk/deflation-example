# Scale, trajectory and operating-condition records

These files hold every timed run behind the scale, trajectory and
operating-condition comparisons of the prescribed-flow CHT study. The
[study page](../../docs/prescribed-speedup-v20.md) explains the comparisons,
definitions and reproduction commands.

| File | Content |
| --- | --- |
| [`evidence.tar.gz`](evidence.tar.gz) | One digest per run with the SHA-256 of its source record, the study protocol, the operating scenarios, the job list and the time-refinement checks |
| [`populations.md`](populations.md) | Every comparison with converged and attempted repetitions per method and the resulting speedup |
| [`rows.json`](rows.json) | The table rows and their comparison keys |
| [`summary.json`](summary.json) | Per-method statistics of every comparison |
| [`manifest.json`](manifest.json) | Checksums of these files and the source commits |
