import { setPipelineWorkerUrl, setPipelinesBaseUrl } from 'itk-wasm';

const pipelineWorkerUrl = '/itk/web-workers/itk-wasm-pipeline.min.worker.js';
setPipelineWorkerUrl(pipelineWorkerUrl);
const pipelineBaseUrl = '/itk/pipelines';
setPipelinesBaseUrl(pipelineBaseUrl);

import '@shoelace-style/shoelace/dist/themes/light.css';
import { setBasePath } from '@shoelace-style/shoelace/dist/utilities/base-path.js';
setBasePath('dist/shoelace');
