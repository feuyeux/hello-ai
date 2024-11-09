# openwebui

https://docs.openwebui.com/

```sh
docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda

```

```sh
http://localhost:3000/
```
----

```sh
docker run --hostname=c59c988ce374 --user=0:0 --mac-address=02:42:ac:11:00:02 --env=PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env=LANG=C.UTF-8 --env=GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D --env=PYTHON_VERSION=3.11.9 --env=PYTHON_PIP_VERSION=24.0 --env=PYTHON_SETUPTOOLS_VERSION=65.5.1 --env=PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/dbf0c85f76fb6e1ab42aa672ffca6f0a675d9ee4/public/get-pip.py --env=PYTHON_GET_PIP_SHA256=dfe9fd5c28dc98b5ac17979a953ea550cec37ae1b47a5116007395bfacff2ab9 --env=ENV=prod --env=PORT=8080 --env=USE_OLLAMA_DOCKER=false --env=USE_CUDA_DOCKER=true --env=USE_CUDA_DOCKER_VER=cu121 --env=USE_EMBEDDING_MODEL_DOCKER=sentence-transformers/all-MiniLM-L6-v2 --env=USE_RERANKING_MODEL_DOCKER= --env=OLLAMA_BASE_URL=/ollama --env=OPENAI_API_BASE_URL= --env=OPENAI_API_KEY= --env=WEBUI_SECRET_KEY= --env=SCARF_NO_ANALYTICS=true --env=DO_NOT_TRACK=true --env=ANONYMIZED_TELEMETRY=false --env=WHISPER_MODEL=base --env=WHISPER_MODEL_DIR=/app/backend/data/cache/whisper/models --env=RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 --env=RAG_RERANKING_MODEL= --env=SENTENCE_TRANSFORMERS_HOME=/app/backend/data/cache/embedding/models --env=HF_HOME=/app/backend/data/cache/embedding/models --env=HOME=/root --env=WEBUI_BUILD_VERSION=162643a4b17edc24426980e4eddcdcfd7c3f7970 --volume=open-webui:/app/backend/data --network=bridge --workdir=/app/backend -p 3000:8080 --restart=always --label='org.opencontainers.image.created=2024-06-13T18:15:05.847Z' --label='org.opencontainers.image.description=User-friendly WebUI for LLMs (Formerly Ollama WebUI)' --label='org.opencontainers.image.licenses=MIT' --label='org.opencontainers.image.revision=162643a4b17edc24426980e4eddcdcfd7c3f7970' --label='org.opencontainers.image.source=https://github.com/open-webui/open-webui' --label='org.opencontainers.image.title=open-webui' --label='org.opencontainers.image.url=https://github.com/open-webui/open-webui' --label='org.opencontainers.image.version=main-cuda' --add-host host.docker.internal:host-gateway --runtime=runc -d ghcr.io/open-webui/open-webui:cuda
```