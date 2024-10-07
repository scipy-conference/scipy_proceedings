import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  build: {
    outDir: "dist",
  },
  server: {
    port: 5173,
  },
  optimizeDeps: {
    exclude: ["@itk-wasm/htj2k", "itk-wasm", "@itk-viewer/io", "eventemitter3", "p-queue"],
  },
  plugins: [
    // collect lazy loaded JavaScript and Wasm bundles in public directory
    viteStaticCopy({
      targets: [
        {
          src: "./node_modules/itk-wasm/dist/pipeline/web-workers/bundles/itk-wasm-pipeline.min.worker.js",
          dest: "itk/web-workers",
        },
        {
          src: "node_modules/.pnpm/@itk-viewer+blosc-zarr@0.2.3/node_modules/@itk-viewer/blosc-zarr/emscripten-build/*",
          dest: "itk/pipelines",
        },
        {
          src: "./node_modules/@shoelace-style/shoelace/dist/assets",
          dest: "dist/shoelace",
        },
      ],
    }),
  ],
});
