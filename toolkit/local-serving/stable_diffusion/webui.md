# STABLE DIFFUSION WEBUI

<https://github.com/AUTOMATIC1111/stable-diffusion-webui>

```sh
git clone https://github.com/AbdBarho/stable-diffusion-webui-docker.git
cd /d/coding/stable-diffusion-webui-docker
```

```sh
code services/AUTOMATIC1111/Dockerfile

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN sed -i 's@http://security.ubuntu.com/ubuntu@https://mirrors.ustc.edu.cn/ubuntu@g' /etc/apt/sources.list
RUN sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.ustc.edu.cn/ubuntu@g' /etc/apt/sources.list
```

```sh
code docker-compose.yml

  auto: &automatic
...
    environment:
      - CLI_ARGS=--allow-code --medvram --xformers --enable-insecure-extension-access --listen --api
    ports:
      - 7860:7860
```

```sh
docker compose --profile download up --build
# wait until its done, then:
# docker compose --profile [ui] up --build
# where [ui] is one of: invoke | auto | auto-cpu | comfy | comfy-cpu
docker compose --profile auto up --build
```

```sh
cd /d/coding/stable-diffusion-webui-docker
docker compose --profile auto up --build

http://localhost:7860/
```

```sh
http://localhost:3000/

# openui -- (7860) --> image model --> stable-diffusion-webui
ip=$(ipconfig | grep -oP 'IPv4 Address.* : \K(\d{1,3}\.){3}\d{1,3}' | awk 'NR==1{print $1}')
echo "http://$ip:7860/"
```

Describe a man in a dog suit. This is for a stable diffusion prompt.