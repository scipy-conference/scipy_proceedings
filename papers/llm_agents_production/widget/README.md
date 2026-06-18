# Agent Optimization Widget

This folder contains a static decision-support widget for the SciPy 2026 paper. It is designed to be opened directly in a browser and does not call any hosted model, search service, or internal API.

## Open The Widget

Open `index.html` in a browser:

```powershell
Start-Process .\index.html
```

The widget embeds aggregate calibration values mirrored from:

- `../widget_data/single_agent_optimization_results.csv`
- `../widget_data/single_agent_quality_probe.csv`
- `../widget_data/multi_agent_operating_points.csv`

The values are embedded in `app.js` so the widget can be opened directly from the filesystem without a local web server or browser permissions for loading adjacent CSV files.

No raw source text, prompts, retrieved evidence, per-record outputs, logs, or checkpoints are included.

## Intended Use

Use the widget to compare observed operating points:

- Single-agent worker-count and API-concurrency settings.
- Multi-agent known-solvable concurrency runs.
- Verifier concurrency caps.
- Token-pressure stress settings.

The results are calibration measurements from specific internal experiments. They should guide configuration discussion, not replace a fresh validation run on a new authorized workload.
