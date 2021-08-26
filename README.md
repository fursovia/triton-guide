# Triton Guide

```bash
docker run -itd --rm -v /mnt/chatbot_models2/fursov/tasks/triton-guide/model_registry:/models \
  -p 8040:8000 -p 8041:8001 -p 8042:8002 \
  --name triton docker-proxy.tcsbank.ru/nvidia/tritonserver:20.09-py3 \
  tritonserver --model-repository /models --log-verbose 4
```
