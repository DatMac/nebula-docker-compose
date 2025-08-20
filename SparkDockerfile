FROM bitnami/spark:2.4.3

USER root

RUN echo "deb http://archive.debian.org/debian/ stretch main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security stretch/updates main" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends python3 python3-pip

RUN pip3 install \
    dataclasses \
    "Faker==15.3.4" \
    "typing-extensions==4.1.1"

RUN groupadd --system --gid 1001 sparky && \
    useradd --system --uid 1001 --gid sparky --shell /bin/bash --create-home sparky && \
    chown -R sparky:sparky /opt/bitnami/spark

USER sparky
