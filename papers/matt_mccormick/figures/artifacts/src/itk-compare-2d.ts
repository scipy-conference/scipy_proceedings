import * as lit from "lit";
import { customElement } from "lit/decorators.js";
import { Ref, createRef, ref } from "lit/directives/ref.js";

import "@itk-viewer/element/itk-viewer-element.js";
import { ItkViewer } from "@itk-viewer/element/itk-viewer-element.js";
import "@itk-viewer/element/itk-viewport.js";
import { ItkViewport } from "@itk-viewer/element/itk-viewport.js";
import { ItkView2d } from "@itk-viewer/element/itk-view-2d.js";
import "@itk-viewer/element/itk-view-2d.js";
import "@itk-viewer/element/itk-view-2d-vtkjs.js";
import { ViewControlsShoelace } from "@itk-viewer/element/itk-view-controls-shoelace.js";
import "@itk-viewer/element/itk-view-controls-shoelace.js";
import { Camera } from "@itk-viewer/viewer/camera.js";

import { ActorRefFrom, SnapshotFrom } from "xstate";
import { View2dActor } from "@itk-viewer/viewer/view-2d.js";
import { viewerMachine } from "@itk-viewer/viewer/viewer.js";

class ViewActorForwarder {
  viewLeft: View2dActor;
  viewRight: View2dActor;

  constructor(viewLeft: View2dActor, viewRight: View2dActor) {
    this.viewLeft = viewLeft;
    this.viewRight = viewRight;
  }

  send(event: any) {
    this.viewLeft.send(event);
    this.viewRight.send(event);
  }

  getSnapshot(): SnapshotFrom<View2dActor> {
    return this.viewLeft.getSnapshot();
  }

  subscribe(callback: (snapshot: SnapshotFrom<View2dActor>) => void) {
    this.viewLeft.subscribe(callback);
    this.viewRight.subscribe(callback);
  }
}

@customElement("itk-compare-2d")
export class ItkCompare2d extends lit.LitElement {
  viewerLeft: Ref<ItkViewer> = createRef();
  viewerRight: Ref<ItkViewer> = createRef();
  viewportLeft: Ref<ItkViewport> = createRef();
  viewportRight: Ref<ItkViewport> = createRef();
  viewLeft: Ref<ItkView2d> = createRef();
  viewRight: Ref<ItkView2d> = createRef();
  controls: Ref<ViewControlsShoelace> = createRef();
  viewActorForwarder: ViewActorForwarder | undefined;

  getViewerLeftActor(): ActorRefFrom<typeof viewerMachine> | undefined {
    return this.viewerLeft.value?.getActor();
  }

  getViewerRightActor(): ActorRefFrom<typeof viewerMachine> | undefined {
    return this.viewerRight.value?.getActor();
  }

  getViewLeftActor(): View2dActor | undefined {
    return this.viewLeft.value?.getActor();
  }

  getViewRightActor(): View2dActor | undefined {
    return this.viewRight.value?.getActor();
  }

  render() {
    return lit.html`
      <itk-viewer class="fill" ${ref(
        this.viewerLeft
      )} style="flex-direction: row; gap: 0.25rem">
        <itk-viewport ${ref(this.viewportLeft)} class="fill">
          <div style="position: relative" class="fill">
            <div style="position: absolute; top: 0.25rem; left: 0.25rem">
              <itk-view-controls-shoelace hideColorUi hideTransferFunctionUi
                view="2d"
                ${ref(this.controls)}
              ></itk-view-controls-shoelace>
            </div>
            <itk-view-2d ${ref(this.viewLeft)} class="fill">
              <itk-view-2d-vtkjs></itk-view-2d-vtkjs>
            </itk-view-2d>
          </div>
        </itk-viewport>
      </itk-viewer>
      <itk-viewer class="fill" ${ref(
        this.viewerRight
      )} style="flex-direction: row; gap: 0.25rem">
        <itk-viewport ${ref(this.viewportRight)} class="fill">
            <itk-view-2d ${ref(this.viewRight)} class="fill">
              <itk-view-2d-vtkjs></itk-view-2d-vtkjs>
            </itk-view-2d>
        </itk-viewport>
      </itk-viewer>
    `;
  }

  protected async firstUpdated() {
    // Wait for view to render so view actor is created
    await this.updateComplete;

    this.viewActorForwarder = new ViewActorForwarder(
      this.getViewLeftActor() as View2dActor,
      this.getViewRightActor() as View2dActor
    );
    const controls = this.controls.value!;
    controls.setActor(this.viewActorForwarder as unknown as View2dActor);

    const camera = this.viewportLeft.value!.getActor()?.getSnapshot().context
      .camera as Camera;
    this.viewportRight.value?.getActor()?.send({ type: "setCamera", camera });
  }

  static styles = lit.css`
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
    "itk-compare-2d": ItkCompare2d;
  }
}
