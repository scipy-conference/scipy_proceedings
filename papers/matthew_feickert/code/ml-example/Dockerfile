ARG CUDA_VERSION="12"
ARG ENVIRONMENT="gpu"

FROM ghcr.io/prefix-dev/pixi:noble AS build

# Redeclaring ARGS in a stage without a value inherits the global default
ARG CUDA_VERSION
ARG ENVIRONMENT

WORKDIR /app
COPY . .
ENV CONDA_OVERRIDE_CUDA=$CUDA_VERSION
RUN pixi install --locked --environment $ENVIRONMENT
RUN echo "#!/bin/bash" > /app/entrypoint.sh && \
    pixi shell-hook --environment $ENVIRONMENT -s bash >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh

FROM ghcr.io/prefix-dev/pixi:noble AS final

ARG ENVIRONMENT

WORKDIR /app
COPY --from=build /app/.pixi/envs/$ENVIRONMENT /app/.pixi/envs/$ENVIRONMENT
COPY --from=build /app/pixi.toml /app/pixi.toml
COPY --from=build /app/pixi.lock /app/pixi.lock
# The ignore files are needed for 'pixi run' to work in the container
COPY --from=build /app/.pixi/.gitignore /app/.pixi/.gitignore
COPY --from=build /app/.pixi/.condapackageignore /app/.pixi/.condapackageignore
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
