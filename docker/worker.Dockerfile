FROM python:3.12-slim

RUN groupadd --system app && useradd --system --create-home --gid app app

WORKDIR /workspace

RUN mkdir -p /workspace && chown -R app:app /workspace

USER app

CMD ["python", "-c", "print('se-lab worker ready')"]
