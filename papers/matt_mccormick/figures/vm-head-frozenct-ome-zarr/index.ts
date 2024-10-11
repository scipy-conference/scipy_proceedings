import "./assetPathSetup.js";
import { ZarrMultiscaleSpatialImage } from "@itk-viewer/io/ZarrMultiscaleSpatialImage.js";

import "@itk-viewer/element/itk-viewer-3d.js";

const omeZarrPath = "/vm_head_frozenct.zarr";

const makeZarrImage = (imagePath: string) => {
  const url = new URL(imagePath, document.location.origin);
  return ZarrMultiscaleSpatialImage.fromUrl(url);
};

document.addEventListener("DOMContentLoaded", async function () {
  const image = await makeZarrImage(omeZarrPath);

  const viewerElement = document.querySelector("#viewer");
  if (!viewerElement) throw new Error("Could not find element");
  const viewer = viewerElement.getActor();
  console.log(viewer);
  console.log(viewerElement);
  viewer!.send({ type: "setImage", image, name: "image" });

  const imageActor = viewer!
    .getSnapshot()
    .context.viewports[0].getSnapshot()
    .context.views[0].getSnapshot().context.imageActor;
  imageActor.send({ type: "colorMap", colorMap: "CT-Bone", component: 0 });
  console.log(imageActor);
  const context = viewer!.getSnapshot().context;
  const camera = context.viewports[0].getSnapshot().context.camera;
  console.log(camera);

  camera.send({
    type: "setPose",
    pose: {
      center: new Float32Array([-0.263671875, -30.263671875, -135.5]),
      distance: 612.2755605271669,
      rotation: new Float32Array([
        0.5477936267852783, 0.19993622601032257, 0.2401820719242096,
        0.7760542631149292,
      ]),
    },
  });

  setTimeout(() => {
    imageActor.send({ type: "normalizedOpacityPoints", points: [[0.25, 0.05], [0.5, 0.7], [0.8, 1.0]], component: 0 });
  }, 1500);
});
