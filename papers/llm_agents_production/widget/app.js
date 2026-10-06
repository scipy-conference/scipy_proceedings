const singleAgentRuns = [
  {
    experiment: 'baseline',
    run: 'baseline_limit200_w8_a3_t300_k50',
    records: 200,
    successful: 199,
    completionYield: 0.995,
    workers: 8,
    apiConcurrency: 3,
    runtimeMinutes: 93.65,
    rpm: 2.125,
    avgTime: 28.236,
    totalTokens: 14631268,
    cost: 24.4563,
    costPerRecord: 0.1229,
    notes: 'Initial 200-record baseline',
  },
  {
    experiment: 'worker_sweep',
    run: 'worker_limit200_w3_a3_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 3,
    apiConcurrency: 3,
    runtimeMinutes: 63.35,
    rpm: 3.157,
    avgTime: 19.005,
    totalTokens: 14401763,
    cost: 23.7347,
    costPerRecord: 0.1187,
    notes: 'Worker sweep with fixed API concurrency 3',
  },
  {
    experiment: 'worker_sweep',
    run: 'worker_limit200_w4_a3_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 4,
    apiConcurrency: 3,
    runtimeMinutes: 62.68,
    rpm: 3.191,
    avgTime: 18.805,
    totalTokens: 14489242,
    cost: 24.0548,
    costPerRecord: 0.1203,
    notes: 'Worker sweep with fixed API concurrency 3',
  },
  {
    experiment: 'worker_sweep',
    run: 'worker_limit200_w6_a3_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 6,
    apiConcurrency: 3,
    runtimeMinutes: 63.1,
    rpm: 3.17,
    avgTime: 18.93,
    totalTokens: 14428523,
    cost: 23.845,
    costPerRecord: 0.1192,
    notes: 'Worker sweep with fixed API concurrency 3',
  },
  {
    experiment: 'worker_sweep',
    run: 'worker_limit200_w8_a3_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 8,
    apiConcurrency: 3,
    runtimeMinutes: 61.58,
    rpm: 3.248,
    avgTime: 18.475,
    totalTokens: 14364758,
    cost: 23.7088,
    costPerRecord: 0.1185,
    notes: 'Worker sweep with fixed API concurrency 3',
  },
  {
    experiment: 'worker_sweep',
    run: 'worker_limit200_w12_a3_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 3,
    runtimeMinutes: 59.3,
    rpm: 3.373,
    avgTime: 17.79,
    totalTokens: 14414866,
    cost: 23.9126,
    costPerRecord: 0.1196,
    notes: 'Worker sweep with fixed API concurrency 3',
  },
  {
    experiment: 'api_sweep',
    run: 'api_limit200_w12_a1_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 1,
    runtimeMinutes: 168.25,
    rpm: 1.189,
    avgTime: 50.4,
    totalTokens: 14449655,
    cost: 24.0015,
    costPerRecord: 0.12,
    notes: 'API sweep with fixed worker count 12',
  },
  {
    experiment: 'api_sweep',
    run: 'api_limit200_w12_a2_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 2,
    runtimeMinutes: 81.9,
    rpm: 2.442,
    avgTime: 24.5,
    totalTokens: 14440874,
    cost: 23.9092,
    costPerRecord: 0.1195,
    notes: 'API sweep with fixed worker count 12',
  },
  {
    experiment: 'api_sweep',
    run: 'api_limit200_w12_a3_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 3,
    runtimeMinutes: 53.95,
    rpm: 3.707,
    avgTime: 16.2,
    totalTokens: 14386883,
    cost: 23.7138,
    costPerRecord: 0.1186,
    notes: 'API sweep with fixed worker count 12',
  },
  {
    experiment: 'api_sweep',
    run: 'api_limit200_w12_a4_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 4,
    runtimeMinutes: 51.82,
    rpm: 3.86,
    avgTime: 15.545,
    totalTokens: 14416249,
    cost: 23.7789,
    costPerRecord: 0.1189,
    notes: 'API sweep with fixed worker count 12',
  },
  {
    experiment: 'api_sweep',
    run: 'api_limit200_w12_a6_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 6,
    runtimeMinutes: 38.23,
    rpm: 5.231,
    avgTime: 11.47,
    totalTokens: 14439306,
    cost: 23.9063,
    costPerRecord: 0.1195,
    notes: 'API sweep with fixed worker count 12',
  },
  {
    experiment: 'api_sweep',
    run: 'api_limit200_w12_a8_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 8,
    runtimeMinutes: 27.45,
    rpm: 7.286,
    avgTime: 8.235,
    totalTokens: 14587564,
    cost: 24.2628,
    costPerRecord: 0.1213,
    notes: 'API sweep with fixed worker count 12',
  },
  {
    experiment: 'api_sweep',
    run: 'api_limit200_w12_a12_t300_k50',
    records: 200,
    successful: 200,
    completionYield: 1.0,
    workers: 12,
    apiConcurrency: 12,
    runtimeMinutes: 19.02,
    rpm: 10.517,
    avgTime: 5.705,
    totalTokens: 14437047,
    cost: 24.0321,
    costPerRecord: 0.1202,
    notes: 'Best observed single-agent throughput',
  },
];

const multiAgentRuns = [
  {
    family: 'known_solvable_concurrency',
    run: 'known_solvable_c2',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 2,
    verifierConcurrency: null,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.96,
    exactId: 0.93,
    exactText: 0.34,
    exactTriplet: null,
    consistency: 0.93,
    rpm: 2.047,
    peakTpm: 67386,
    reasoningTokens: 597283,
    costPer100: 4.9877,
    pareto: false,
    recommended: false,
    notes:
      'Known-solvable cohort selected from prior exact label-ID successes.',
  },
  {
    family: 'known_solvable_concurrency',
    run: 'known_solvable_c6',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 6,
    verifierConcurrency: null,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.973,
    exactId: 0.927,
    exactText: 0.32,
    exactTriplet: null,
    consistency: 0.93,
    rpm: 6.086,
    peakTpm: 153087,
    reasoningTokens: 604667,
    costPer100: 4.9752,
    pareto: false,
    recommended: false,
    notes:
      'Known-solvable cohort selected from prior exact label-ID successes.',
  },
  {
    family: 'known_solvable_concurrency',
    run: 'known_solvable_c12',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 12,
    verifierConcurrency: null,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.96,
    exactId: 0.937,
    exactText: 0.343,
    exactTriplet: null,
    consistency: 0.96,
    rpm: 8.916,
    peakTpm: 269573,
    reasoningTokens: 604789,
    costPer100: 5.0116,
    pareto: false,
    recommended: false,
    notes:
      'Known-solvable cohort selected from prior exact label-ID successes.',
  },
  {
    family: 'verifier_cap_sweep',
    run: 'verifier_cap_1',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 10,
    verifierConcurrency: 1,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.863,
    exactId: 0.807,
    exactText: null,
    exactTriplet: 0.807,
    consistency: 0.81,
    rpm: 9.69,
    peakTpm: 328727,
    reasoningTokens: 529444,
    costPer100: 5.6346,
    pareto: false,
    recommended: false,
    notes: 'Verifier cap sweep on the known-solvable cohort.',
  },
  {
    family: 'verifier_cap_sweep',
    run: 'verifier_cap_3',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 10,
    verifierConcurrency: 3,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.89,
    exactId: 0.84,
    exactText: null,
    exactTriplet: 0.84,
    consistency: 0.91,
    rpm: 10.303,
    peakTpm: 392915,
    reasoningTokens: 529068,
    costPer100: 5.6326,
    pareto: true,
    recommended: true,
    notes:
      'Recommended balanced point: strongest repeat consistency with a bounded verifier cap.',
  },
  {
    family: 'verifier_cap_sweep',
    run: 'verifier_cap_6',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 10,
    verifierConcurrency: 6,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.86,
    exactId: 0.803,
    exactText: null,
    exactTriplet: 0.803,
    consistency: 0.83,
    rpm: 10.336,
    peakTpm: 412171,
    reasoningTokens: 520310,
    costPer100: 5.6131,
    pareto: false,
    recommended: false,
    notes: 'Verifier cap sweep on the known-solvable cohort.',
  },
  {
    family: 'verifier_cap_sweep',
    run: 'verifier_cap_10',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 10,
    verifierConcurrency: 10,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.88,
    exactId: 0.843,
    exactText: null,
    exactTriplet: 0.843,
    consistency: 0.86,
    rpm: 10.506,
    peakTpm: 428031,
    reasoningTokens: 516637,
    costPer100: 5.5884,
    pareto: true,
    recommended: false,
    notes:
      'Cost/throughput edge, with wider verifier fanout than the recommended point.',
  },
  {
    family: 'verifier_cap_sweep',
    run: 'verifier_cap_20',
    records: 100,
    recordRuns: 300,
    batchConcurrency: 10,
    verifierConcurrency: 20,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.88,
    exactId: 0.833,
    exactText: null,
    exactTriplet: 0.833,
    consistency: 0.82,
    rpm: 9.911,
    peakTpm: 389010,
    reasoningTokens: 520110,
    costPer100: 5.6144,
    pareto: false,
    recommended: false,
    notes: 'Verifier cap sweep on the known-solvable cohort.',
  },
  {
    family: 'token_pressure_stress',
    run: 'baseline_medium_c10_v3',
    records: 100,
    recordRuns: 100,
    batchConcurrency: 10,
    verifierConcurrency: 3,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.88,
    exactId: 0.82,
    exactText: null,
    exactTriplet: 0.82,
    consistency: null,
    rpm: 10.184,
    peakTpm: 363982,
    reasoningTokens: 175989,
    costPer100: 5.6142,
    pareto: false,
    recommended: false,
    notes: 'Medium-reasoning baseline.',
  },
  {
    family: 'token_pressure_stress',
    run: 'high_reasoning_c10_v3',
    records: 100,
    recordRuns: 100,
    batchConcurrency: 10,
    verifierConcurrency: 3,
    reasoningEffort: 'high',
    extraClassifier: false,
    predictionYield: 0.88,
    exactId: 0.83,
    exactText: null,
    exactTriplet: 0.83,
    consistency: null,
    rpm: 7.841,
    peakTpm: 287838,
    reasoningTokens: 312143,
    costPer100: 7.009,
    pareto: false,
    recommended: false,
    notes: 'High-reasoning confirmation run.',
  },
  {
    family: 'token_pressure_stress',
    run: 'high_concurrency_medium_c20_v10',
    records: 100,
    recordRuns: 100,
    batchConcurrency: 20,
    verifierConcurrency: 10,
    reasoningEffort: 'medium',
    extraClassifier: false,
    predictionYield: 0.86,
    exactId: 0.79,
    exactText: null,
    exactTriplet: 0.79,
    consistency: null,
    rpm: 16.687,
    peakTpm: 706561,
    reasoningTokens: 173163,
    costPer100: 5.6298,
    pareto: true,
    recommended: false,
    notes: 'Latency-biased point with lower strict agreement.',
  },
  {
    family: 'token_pressure_stress',
    run: 'max_pressure_high_c20_v20_all_on',
    records: 100,
    recordRuns: 100,
    batchConcurrency: 20,
    verifierConcurrency: 20,
    reasoningEffort: 'high',
    extraClassifier: true,
    predictionYield: 0.89,
    exactId: 0.84,
    exactText: null,
    exactTriplet: 0.84,
    consistency: null,
    rpm: 13.71,
    peakTpm: 576070,
    reasoningTokens: 308303,
    costPer100: 6.9536,
    pareto: true,
    recommended: false,
    notes: 'High reasoning plus extra classifier stage.',
  },
];

const els = {
  viewButtons: document.querySelectorAll('[data-view-button]'),
  panels: document.querySelectorAll('[data-control-panel]'),
  singleWorker: document.getElementById('singleWorker'),
  singleApi: document.getElementById('singleApi'),
  singleRecords: document.getElementById('singleRecords'),
  singleMinYield: document.getElementById('singleMinYield'),
  singleMinYieldOut: document.getElementById('singleMinYieldOut'),
  singleMinThroughput: document.getElementById('singleMinThroughput'),
  singleMinThroughputOut: document.getElementById('singleMinThroughputOut'),
  singleMaxCost: document.getElementById('singleMaxCost'),
  multiFamily: document.getElementById('multiFamily'),
  multiRun: document.getElementById('multiRun'),
  multiMinYield: document.getElementById('multiMinYield'),
  multiMinYieldOut: document.getElementById('multiMinYieldOut'),
  multiMinAgreement: document.getElementById('multiMinAgreement'),
  multiMinAgreementOut: document.getElementById('multiMinAgreementOut'),
  multiMinThroughput: document.getElementById('multiMinThroughput'),
  multiMinThroughputOut: document.getElementById('multiMinThroughputOut'),
  multiMaxCost: document.getElementById('multiMaxCost'),
  statusStrip: document.getElementById('statusStrip'),
  metrics: document.getElementById('metrics'),
  throughputChart: document.getElementById('throughputChart'),
  scatterChart: document.getElementById('scatterChart'),
  throughputCaption: document.getElementById('throughputCaption'),
  scatterCaption: document.getElementById('scatterCaption'),
  runTable: document.getElementById('runTable'),
  notes: document.getElementById('notes'),
};

let activeView = 'multi';

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value))
    return 'n/a';
  return `${(value * 100).toFixed(digits)}%`;
}

function number(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value))
    return 'n/a';
  return Number(value).toFixed(digits);
}

function usd(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value))
    return 'n/a';
  return `USD ${Number(value).toFixed(digits)}`;
}

function compact(value) {
  if (value === null || value === undefined || Number.isNaN(value))
    return 'n/a';
  if (value >= 1000000) return `${(value / 1000000).toFixed(2)}M`;
  if (value >= 1000) return `${Math.round(value / 1000)}k`;
  return String(value);
}

function labelize(text) {
  return text.replaceAll('_', ' ');
}

function option(select, value, label) {
  const node = document.createElement('option');
  node.value = value;
  node.textContent = label;
  select.appendChild(node);
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => a - b);
}

function populateControls() {
  uniqueSorted(singleAgentRuns.map((run) => run.workers)).forEach((value) =>
    option(els.singleWorker, value, value),
  );
  uniqueSorted(singleAgentRuns.map((run) => run.apiConcurrency)).forEach(
    (value) => option(els.singleApi, value, value),
  );
  els.singleWorker.value = '12';
  els.singleApi.value = '12';
  els.multiFamily.value = 'verifier_cap_sweep';
  updateMultiRunOptions();
  els.multiRun.value = 'verifier_cap_3';
}

function updateMultiRunOptions() {
  const family = els.multiFamily.value;
  const runs = multiAgentRuns.filter((run) => run.family === family);
  els.multiRun.innerHTML = '';
  runs.forEach((run) => {
    const verifier =
      run.verifierConcurrency === null
        ? ''
        : ` / verifier cap ${run.verifierConcurrency}`;
    option(
      els.multiRun,
      run.run,
      `${labelize(run.run)} (batch ${run.batchConcurrency}${verifier})`,
    );
  });
}

function setActiveView(view) {
  activeView = view;
  els.viewButtons.forEach((button) => {
    const selected = button.dataset.viewButton === view;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-selected', String(selected));
  });
  els.panels.forEach((panel) =>
    panel.classList.toggle('hidden', panel.dataset.controlPanel !== view),
  );
  render();
}

function selectedSingleRun() {
  const worker = Number(els.singleWorker.value);
  const api = Number(els.singleApi.value);
  const exact = singleAgentRuns.filter(
    (run) => run.workers === worker && run.apiConcurrency === api,
  );
  if (exact.length) {
    return exact.sort((a, b) => {
      if (a.experiment === 'baseline' && b.experiment !== 'baseline') return 1;
      if (a.experiment !== 'baseline' && b.experiment === 'baseline') return -1;
      return b.rpm - a.rpm;
    })[0];
  }
  return singleAgentRuns
    .map((run) => ({
      run,
      distance:
        Math.abs(run.workers - worker) + Math.abs(run.apiConcurrency - api),
    }))
    .sort((a, b) => a.distance - b.distance || b.run.rpm - a.run.rpm)[0].run;
}

function selectedMultiRun() {
  return (
    multiAgentRuns.find((run) => run.run === els.multiRun.value) ||
    multiAgentRuns[0]
  );
}

function singleThresholds() {
  return {
    minYield: Number(els.singleMinYield.value) / 100,
    minThroughput: Number(els.singleMinThroughput.value),
    maxCost: Number(els.singleMaxCost.value),
  };
}

function singleFeasible(run, thresholds = singleThresholds()) {
  return (
    run.completionYield >= thresholds.minYield &&
    run.rpm >= thresholds.minThroughput &&
    run.costPerRecord <= thresholds.maxCost
  );
}

function multiThresholds() {
  return {
    minYield: Number(els.multiMinYield.value) / 100,
    minAgreement: Number(els.multiMinAgreement.value) / 100,
    minThroughput: Number(els.multiMinThroughput.value),
    maxCost: Number(els.multiMaxCost.value),
  };
}

function strictAgreement(run) {
  return run.exactTriplet ?? run.exactId;
}

function strictAgreementLabel(run) {
  return run.exactTriplet === null ? 'Exact label-ID' : 'Exact label-triplet';
}

function multiFeasible(run, thresholds = multiThresholds()) {
  return (
    run.predictionYield >= thresholds.minYield &&
    strictAgreement(run) >= thresholds.minAgreement &&
    run.rpm >= thresholds.minThroughput &&
    run.costPer100 <= thresholds.maxCost
  );
}

function bestSingleFeasible(runs) {
  return [...runs].sort(
    (a, b) => b.rpm - a.rpm || a.costPerRecord - b.costPerRecord,
  )[0];
}

function bestMultiFeasible(runs) {
  return [...runs].sort((a, b) => {
    if (a.recommended !== b.recommended)
      return Number(b.recommended) - Number(a.recommended);
    if (a.pareto !== b.pareto) return Number(b.pareto) - Number(a.pareto);
    return (
      strictAgreement(b) - strictAgreement(a) ||
      b.predictionYield - a.predictionYield ||
      b.rpm - a.rpm ||
      a.costPer100 - b.costPer100
    );
  })[0];
}

function metric(label, value, subtext) {
  return `<article class="metric"><span>${label}</span><strong>${value}</strong><small>${subtext}</small></article>`;
}

function pill(label, state) {
  return `<span class="pill ${state}">${label}</span>`;
}

function renderSingle() {
  const run = selectedSingleRun();
  const requestedWorker = Number(els.singleWorker.value);
  const requestedApi = Number(els.singleApi.value);
  const projectedRecords = Math.max(
    1,
    Number(els.singleRecords.value || run.records),
  );
  const projectedCost = run.costPerRecord * projectedRecords;
  const projectedMinutes = projectedRecords / run.rpm;
  const exactConfig =
    run.workers === requestedWorker && run.apiConcurrency === requestedApi;
  const baseline = singleAgentRuns.find(
    (item) => item.experiment === 'baseline',
  );
  const thresholds = singleThresholds();
  const feasibleRows = singleAgentRuns.filter((item) =>
    singleFeasible(item, thresholds),
  );
  const bestFeasible = bestSingleFeasible(feasibleRows);
  const checks = [
    {
      label: `Yield >= ${pct(thresholds.minYield)}`,
      ok: run.completionYield >= thresholds.minYield,
    },
    {
      label: `Throughput >= ${number(thresholds.minThroughput, 1)}/min`,
      ok: run.rpm >= thresholds.minThroughput,
    },
    {
      label: `Cost <= ${usd(thresholds.maxCost, 3)}/record`,
      ok: run.costPerRecord <= thresholds.maxCost,
    },
  ];

  els.statusStrip.innerHTML = [
    pill(
      exactConfig
        ? 'Observed configuration'
        : `Nearest observed: ${run.workers} workers / API concurrency ${run.apiConcurrency}`,
      exactConfig ? 'pass' : 'warn',
    ),
    pill(
      `${feasibleRows.length}/${singleAgentRuns.length} feasible observed configs`,
      feasibleRows.length ? 'pass' : 'fail',
    ),
    bestFeasible
      ? pill(
          `Best feasible: ${bestFeasible.workers} workers / API concurrency ${bestFeasible.apiConcurrency}`,
          'pass',
        )
      : pill('No feasible observed config', 'fail'),
    ...checks.map((check) => pill(check.label, check.ok ? 'pass' : 'fail')),
  ].join('');

  els.metrics.innerHTML = [
    metric('Usable throughput', number(run.rpm, 2), 'records per minute'),
    metric(
      'Completion yield',
      pct(run.completionYield, 2),
      `${run.successful}/${run.records} successful`,
    ),
    metric(
      'Projected cost',
      usd(projectedCost, 2),
      `${projectedRecords} records at ${usd(run.costPerRecord, 4)}/record`,
    ),
    metric(
      'Projected runtime',
      `${number(projectedMinutes, 1)} min`,
      `observed ${number(run.runtimeMinutes, 1)} min for ${run.records}`,
    ),
  ].join('');

  els.throughputCaption.textContent = 'Selected vs initial baseline';
  drawBarChart(
    els.throughputChart,
    [
      { label: 'Baseline', value: baseline.rpm, color: 'secondary' },
      { label: 'Selected', value: run.rpm, color: 'primary' },
    ],
    'records/min',
  );

  els.scatterCaption.textContent = 'API sweep and worker sweep';
  drawScatter(els.scatterChart, singleAgentRuns, run, {
    x: (item) => item.rpm,
    y: (item) => item.completionYield,
    r: (item) => item.costPerRecord * 120,
    xLabel: 'records/min',
    yLabel: 'completion yield',
    formatY: (value) => pct(value, 0),
    xThreshold: thresholds.minThroughput,
    yThreshold: thresholds.minYield,
  });

  renderSingleTable(run, thresholds);
  els.notes.innerHTML = `<p><strong>Interpretation:</strong> In this dataset, hosted API concurrency was the larger single-agent bottleneck. The widget uses observed aggregate runs only; unobserved worker-count and API-concurrency pairs are represented by the nearest measured configuration.</p><p>Cost estimates use public-market token prices and are not enterprise billing statements.</p>`;
}

function renderMulti() {
  const run = selectedMultiRun();
  const familyRuns = multiAgentRuns.filter(
    (item) => item.family === run.family,
  );
  const thresholds = multiThresholds();
  const feasibleRows = familyRuns.filter((item) =>
    multiFeasible(item, thresholds),
  );
  const bestFeasible = bestMultiFeasible(feasibleRows);
  const selectedStrictAgreement = strictAgreement(run);
  const strictLabel = strictAgreementLabel(run);
  const checks = [
    {
      label: `Yield >= ${pct(thresholds.minYield)}`,
      ok: run.predictionYield >= thresholds.minYield,
    },
    {
      label: `${strictLabel} >= ${pct(thresholds.minAgreement)}`,
      ok: selectedStrictAgreement >= thresholds.minAgreement,
    },
    {
      label: `Throughput >= ${number(thresholds.minThroughput, 1)}/min`,
      ok: run.rpm >= thresholds.minThroughput,
    },
    {
      label: `Cost <= ${usd(thresholds.maxCost, 2)}/100`,
      ok: run.costPer100 <= thresholds.maxCost,
    },
  ];

  const frontierBadges = [pill(labelize(run.family), 'pass')];
  frontierBadges.push(
    pill(
      `${feasibleRows.length}/${familyRuns.length} feasible operating points`,
      feasibleRows.length ? 'pass' : 'fail',
    ),
  );
  frontierBadges.push(
    bestFeasible
      ? pill(`Best feasible: ${labelize(bestFeasible.run)}`, 'pass')
      : pill('No feasible operating point', 'fail'),
  );
  if (run.recommended) {
    frontierBadges.push(pill('Recommended balanced point', 'pass'));
  } else if (run.pareto) {
    frontierBadges.push(pill('Pareto-relevant', 'warn'));
  }

  els.statusStrip.innerHTML = [
    ...frontierBadges,
    ...checks.map((check) => pill(check.label, check.ok ? 'pass' : 'fail')),
  ].join('');

  els.metrics.innerHTML = [
    metric(
      'Prediction yield',
      pct(run.predictionYield, 1),
      `${run.records} record cohort`,
    ),
    metric(
      strictLabel,
      pct(selectedStrictAgreement, 1),
      run.consistency === null
        ? 'single completed run'
        : `${pct(run.consistency, 1)} repeat consistency`,
    ),
    metric(
      'Throughput',
      number(run.rpm, 2),
      'records or record-runs per minute',
    ),
    metric(
      'Cost / 100',
      usd(run.costPer100, 2),
      run.peakTpm
        ? `${compact(run.peakTpm)} peak tokens/min`
        : 'public-price approximation',
    ),
  ].join('');

  const best = [...familyRuns].sort((a, b) => b.rpm - a.rpm)[0];
  els.throughputCaption.textContent = 'Selected vs fastest in family';
  drawBarChart(
    els.throughputChart,
    [
      { label: 'Selected', value: run.rpm, color: 'primary' },
      { label: 'Fastest', value: best.rpm, color: 'secondary' },
    ],
    'records/min',
  );

  els.scatterCaption.textContent = familyTitle(run.family);
  drawScatter(els.scatterChart, familyRuns, run, {
    x: (item) => item.rpm,
    y: (item) => item.exactTriplet ?? item.exactId,
    r: (item) => (item.peakTpm ? Math.max(5, item.peakTpm / 50000) : 7),
    xLabel: 'records/min',
    yLabel: strictLabel.toLowerCase(),
    formatY: (value) => pct(value, 0),
    xThreshold: thresholds.minThroughput,
    yThreshold: thresholds.minAgreement,
  });

  renderMultiTable(familyRuns, run, thresholds);
  els.notes.innerHTML = `<p><strong>Interpretation:</strong> These multi-agent rows are known-solvable regression and pressure tests, not unbiased accuracy estimates. The recommended point uses a verifier cap of 3 because it keeps verifier fanout bounded while preserving high prediction yield, strict agreement, and repeat consistency.</p><p>Pareto labels are computed over comparable exact label-triplet rows using cost per usable record, throughput, prediction yield, and strict agreement. The widget does not launch LLM calls and does not expose sensitive source artifacts.</p>`;
}

function familyTitle(family) {
  if (family === 'known_solvable_concurrency')
    return 'Known-solvable concurrency';
  if (family === 'verifier_cap_sweep') return 'Verifier cap sweep';
  return 'Token-pressure stress';
}

function drawBarChart(svg, bars, unit) {
  const max = Math.max(...bars.map((bar) => bar.value), 1);
  const width = 520;
  const height = 240;
  const left = 52;
  const right = 24;
  const bottom = 42;
  const top = 18;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const barWidth = 96;
  const gap = 68;
  const startX =
    left + (plotWidth - bars.length * barWidth - (bars.length - 1) * gap) / 2;
  let out = `<line class="axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"></line>`;
  bars.forEach((bar, index) => {
    const h = (bar.value / max) * (plotHeight - 16);
    const x = startX + index * (barWidth + gap);
    const y = height - bottom - h;
    const className = bar.color === 'secondary' ? 'bar secondary' : 'bar';
    out += `<rect class="${className}" x="${x}" y="${y}" width="${barWidth}" height="${h}" rx="5"></rect>`;
    out += `<text class="value-label" x="${x + barWidth / 2}" y="${Math.max(14, y - 8)}" text-anchor="middle">${number(bar.value, 2)}</text>`;
    out += `<text class="label" x="${x + barWidth / 2}" y="${height - 16}" text-anchor="middle">${bar.label}</text>`;
  });
  out += `<text class="label" x="${left}" y="16">${unit}</text>`;
  svg.innerHTML = out;
}

function drawScatter(svg, rows, selected, config) {
  const width = 520;
  const height = 240;
  const left = 54;
  const right = 22;
  const top = 18;
  const bottom = 44;
  const xs = rows.map(config.x);
  const ys = rows.map(config.y);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 1);
  const yMin = Math.min(...ys, 0.75);
  const yMax = Math.max(...ys, 1);
  const xPad = (xMax - xMin) * 0.08 || 1;
  const yPad = (yMax - yMin) * 0.12 || 0.05;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const xLow = xMin - xPad;
  const xHigh = xMax + xPad;
  const yLow = yMin - yPad;
  const yHigh = yMax + yPad;
  const scaleX = (value) =>
    left + ((value - xLow) / (xHigh - xLow)) * plotWidth;
  const scaleY = (value) =>
    height - bottom - ((value - yLow) / (yHigh - yLow)) * plotHeight;
  let out = '';
  out += `<line class="axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"></line>`;
  out += `<line class="axis" x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}"></line>`;
  out += `<text class="label" x="${width - right}" y="${height - 12}" text-anchor="end">${config.xLabel}</text>`;
  out += `<text class="label" x="${left}" y="14">${config.yLabel}</text>`;
  if (
    Number.isFinite(config.xThreshold) &&
    config.xThreshold >= xLow &&
    config.xThreshold <= xHigh
  ) {
    const x = scaleX(config.xThreshold);
    out += `<line class="threshold" x1="${x}" y1="${top}" x2="${x}" y2="${height - bottom}"></line>`;
    out += `<text class="threshold-label" x="${x + 5}" y="${top + 14}">min throughput</text>`;
  }
  if (
    Number.isFinite(config.yThreshold) &&
    config.yThreshold >= yLow &&
    config.yThreshold <= yHigh
  ) {
    const y = scaleY(config.yThreshold);
    out += `<line class="threshold" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}"></line>`;
    out += `<text class="threshold-label" x="${width - right - 5}" y="${y - 6}" text-anchor="end">min target</text>`;
  }
  rows.forEach((row) => {
    const x = scaleX(config.x(row));
    const y = scaleY(config.y(row));
    const r = Math.min(14, Math.max(5, config.r(row)));
    const selectedClass = row.run === selected.run ? ' selected' : '';
    out += `<circle class="dot${selectedClass}" cx="${x}" cy="${y}" r="${r}"></circle>`;
    if (row.run === selected.run) {
      const nearRightEdge = x + r + 92 > width - right;
      const labelX = nearRightEdge ? x - r - 7 : x + r + 7;
      const labelAnchor = nearRightEdge ? 'end' : 'start';
      out += `<text class="value-label" x="${labelX}" y="${y - 8}" text-anchor="${labelAnchor}">${number(config.x(row), 2)}, ${config.formatY(config.y(row))}</text>`;
    }
  });
  svg.innerHTML = out;
}

function renderSingleTable(selected, thresholds) {
  const rows = singleAgentRuns
    .map(
      (run) => `
    <tr class="${[run.run === selected.run ? 'selected-row' : '', singleFeasible(run, thresholds) ? 'feasible-row' : 'infeasible-row'].join(' ')}">
      <td>${labelize(run.experiment)}</td>
      <td>${run.workers} workers / API concurrency ${run.apiConcurrency}</td>
      <td>${singleFeasible(run, thresholds) ? 'Yes' : 'No'}</td>
      <td>${number(run.rpm, 2)}</td>
      <td>${pct(run.completionYield, 1)}</td>
      <td>${number(run.runtimeMinutes, 1)}</td>
      <td>${usd(run.costPerRecord, 4)}</td>
      <td>${compact(run.totalTokens)}</td>
    </tr>
  `,
    )
    .join('');
  els.runTable.innerHTML = `
    <thead>
      <tr><th>Experiment</th><th>Config</th><th>Meets constraints</th><th>Records/min</th><th>Yield</th><th>Runtime min</th><th>Cost/record</th><th>Tokens</th></tr>
    </thead>
    <tbody>${rows}</tbody>
  `;
}

function renderMultiTable(rows, selected, thresholds) {
  const body = rows
    .map((run) => {
      const strict = strictAgreement(run);
      const frontier = run.recommended
        ? 'Recommended'
        : run.pareto
          ? 'Pareto'
          : run.exactTriplet === null
            ? 'n/a'
            : 'Dominated';
      const concurrency =
        run.verifierConcurrency === null
          ? `${run.batchConcurrency} batch`
          : `${run.batchConcurrency} batch / ${run.verifierConcurrency} verifier cap`;
      return `
      <tr class="${[run.run === selected.run ? 'selected-row' : '', multiFeasible(run, thresholds) ? 'feasible-row' : 'infeasible-row'].join(' ')}">
        <td>${labelize(run.run)}</td>
        <td>${frontier}</td>
        <td>${multiFeasible(run, thresholds) ? 'Yes' : 'No'}</td>
        <td>${concurrency}</td>
        <td>${run.reasoningEffort}${run.extraClassifier ? ' + classifier' : ''}</td>
        <td>${number(run.rpm, 2)}</td>
        <td>${pct(run.predictionYield, 1)}</td>
        <td>${pct(strict, 1)}</td>
        <td>${run.consistency === null ? 'n/a' : pct(run.consistency, 1)}</td>
        <td>${run.peakTpm === null ? 'n/a' : compact(run.peakTpm)}</td>
        <td>${usd(run.costPer100, 2)}</td>
      </tr>
    `;
    })
    .join('');
  els.runTable.innerHTML = `
    <thead>
      <tr><th>Run</th><th>Frontier</th><th>Meets constraints</th><th>Concurrency</th><th>Reasoning</th><th>Records/min</th><th>Yield</th><th>Strict match</th><th>Consistency</th><th>Peak tokens/min</th><th>Cost/100</th></tr>
    </thead>
    <tbody>${body}</tbody>
  `;
}

function render() {
  els.singleMinYieldOut.textContent = `${number(els.singleMinYield.value, 1)}%`;
  els.singleMinThroughputOut.textContent = number(
    els.singleMinThroughput.value,
    1,
  );
  els.multiMinYieldOut.textContent = `${number(els.multiMinYield.value, 1)}%`;
  els.multiMinAgreementOut.textContent = `${number(els.multiMinAgreement.value, 1)}%`;
  els.multiMinThroughputOut.textContent = number(
    els.multiMinThroughput.value,
    1,
  );
  if (activeView === 'single') {
    renderSingle();
  } else {
    renderMulti();
  }
}

function bind() {
  els.viewButtons.forEach((button) =>
    button.addEventListener('click', () =>
      setActiveView(button.dataset.viewButton),
    ),
  );
  [
    els.singleWorker,
    els.singleApi,
    els.singleRecords,
    els.singleMinYield,
    els.singleMinThroughput,
    els.singleMaxCost,
    els.multiRun,
    els.multiMinYield,
    els.multiMinAgreement,
    els.multiMinThroughput,
    els.multiMaxCost,
  ].forEach((node) => node.addEventListener('input', render));
  els.multiFamily.addEventListener('input', () => {
    updateMultiRunOptions();
    render();
  });
}

populateControls();
bind();
render();
