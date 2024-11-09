# Dify

<https://dify.ai/>
<https://github.com/langgenius/dify>
Define + Modify
通过 Dify 构建 MVP（最小可用产品）获得投资，或通过 POC（概念验证）赢得了客户的订单

<https://docs.dify.ai/v/zh-hans/getting-started/install-self-hosted/docker-compose>

## run dify

```sh
git clone https://github.com/langgenius/dify.git

cd docker
docker compose up -d
```

访问dify

<http://localhost/>

## 配置ollma

### run ollma

```sh
OLLAMA_HOST=$(ipconfig getifaddr en0) ollama serve
```

### 在dify中关联ollma

```sh
echo "http://$(ipconfig getifaddr en0):11434"
```
