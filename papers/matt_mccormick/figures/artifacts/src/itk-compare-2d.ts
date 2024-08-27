import { LitElement, css, html } from 'lit';
import { customElement } from 'lit/decorators.js';
import { Ref, createRef, ref } from 'lit/directives/ref.js';

import "@itk-viewer/element/itk-viewer-element.js";
import { ItkViewer } from "@itk-viewer/element/itk-viewer-element.js";
import "@itk-viewer/element/itk-viewport.js";
import { ItkView2d } from "@itk-viewer/element/itk-view-2d.js";
import "@itk-viewer/element/itk-view-2d.js";
import "@itk-viewer/element/itk-view-2d-vtkjs.js";
import { ViewControlsShoelace } from "@itk-viewer/element/itk-view-controls-shoelace.js";
import "@itk-viewer/element/itk-view-controls-shoelace.js";

import { ActorRefFrom } from 'xstate';
import { viewerMachine } from '@itk-viewer/viewer/viewer.js';


@customElement('itk-compare-2d')
export class ItkCompare2d extends LitElement {
  viewer: Ref<ItkViewer> = createRef();
  view: Ref<ItkView2d> = createRef();
  controls: Ref<ViewControlsShoelace> = createRef();

  getActor(): ActorRefFrom<typeof viewerMachine> | undefined {
    return this.viewer.value?.getActor();
  }

  render() {
    return html`
      <itk-viewer class="fill" ${ref(this.viewer)}>
        <itk-viewport class="fill">
          <div style="position: relative" class="fill">
            <div style="position: absolute; top: 0.25rem; left: 0.25rem">
              <itk-view-controls-shoelace
                view="2d"
                ${ref(this.controls)}
              ></itk-view-controls-shoelace>
            </div>
            <itk-view-2d ${ref(this.view)} class="fill">
              <itk-view-2d-vtkjs></itk-view-2d-vtkjs>
            </itk-view-2d>
          </div>
        </itk-viewport>
      </itk-viewer>
    `;
  }

  protected async firstUpdated() {
    // Wait for view to render so view actor is created
    await this.updateComplete;
    const viewActor = this.view.value!.getActor()!;
    const controls = this.controls.value!;
    controls.setActor(viewActor);
  }

  static styles = css`
    :host {
      width: 100%;
      height: 100%;
      display: flex;
    }

    .fill {
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
  `;
}

declare global {
  interface HTMLElementTagNameMap {
    'itk-compare-2d': ItkCompare2d;
  }
}
