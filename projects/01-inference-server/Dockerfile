# Two-stage: the GPU image for real serving, the CPU image for the simulator.
# Keeping both in one file means the k8s manifests differ only by image tag.
ARG MODE=gpu

FROM vllm/vllm-openai:latest AS gpu
COPY packages/llmkit /opt/llmkit
RUN pip install --no-cache-dir -e /opt/llmkit
COPY projects/01-inference-server/serve_vllm.sh /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve
ENTRYPOINT ["/usr/local/bin/serve"]

FROM python:3.12-slim AS sim
WORKDIR /app
COPY packages/llmkit /app/packages/llmkit
RUN pip install --no-cache-dir -e /app/packages/llmkit fastapi "uvicorn[standard]"
COPY projects/01-inference-server/mock_server.py /app/
EXPOSE 8000
ENTRYPOINT ["python", "mock_server.py", "--host", "0.0.0.0"]

FROM ${MODE} AS final
