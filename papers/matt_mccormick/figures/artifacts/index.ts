import "./assetPathSetup.js";
import { ZarrMultiscaleSpatialImage } from "@itk-viewer/io/ZarrMultiscaleSpatialImage.js";

const aliasedPath = "/aliased.ome.zarr";
const antiAliasedPath = "/antialias.ome.zarr";

const makeZarrImage = (imagePath: string) => {
  const url = new URL(imagePath, document.location.origin);
  return ZarrMultiscaleSpatialImage.fromUrl(url);
};

document.addEventListener("DOMContentLoaded", async function () {
  const aliasedImage = await makeZarrImage(aliasedPath);
  const antiAliasedImage = await makeZarrImage(antiAliasedPath);

  const viewerElement = document.querySelector("#viewer");
  if (!viewerElement) throw new Error("Could not find element");

  const viewerLeft = viewerElement.getViewerLeftActor();
  viewerLeft!.send({ type: "setImage", image: aliasedImage, name: "image" });

  const viewerRight = viewerElement.getViewerRightActor();
  viewerRight!.send({
    type: "setImage",
    image: antiAliasedImage,
    name: "image",
  });
});
