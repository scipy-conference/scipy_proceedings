# SciPy 2026 Optimization Widget Data

This folder contains consolidated aggregate data for the optimization widget.

- `single_agent_optimization_results.csv` contains observed 200-record operational runs.
- `single_agent_quality_probe.csv` contains the 12-record labeled calibration probe.
- `multi_agent_operating_points.csv` contains aggregate multi-agent known-solvable, verifier-cap, token-pressure, and observed Pareto operating points.
- `widget_summary.json` contains compact defaults and recommended display values for the widget.

`../scripts/consolidate_multi_agent_operating_points.py` rebuilds the multi-agent CSV from internal summary tables and usage logs. The generated file keeps only aggregate metrics, public-price cost estimates, and Pareto/recommendation flags.

Cost estimates are market approximations based on publicly available OpenAI GPT-5 pricing:

- GPT-5 input: USD 1.25 per 1M tokens
- GPT-5 output: USD 10.00 per 1M tokens
- text-embedding-3-small: USD 0.02 per 1M tokens

These estimates are not Azure OpenAI enterprise billing and do not include negotiated rates, cached-input discounts, reserved capacity, data residency adjustments, or organization-specific commercial terms.

The files in this folder contain no raw source narratives, prompts, retrieved evidence, per-record outputs, raw logs, or checkpoints.
