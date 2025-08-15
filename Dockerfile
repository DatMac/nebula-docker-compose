FROM bitnami/spark:3.0.0

USER root

RUN groupadd --system --gid 1001 sparky && \
    useradd --system --uid 1001 --gid sparky --shell /bin/bash --create-home sparky && \
    chown -R sparky:sparky /opt/bitnami/spark

USER sparky