# [Host ALL your AI locally](https://www.youtube.com/watch?v=Wjrdr0NU4Sk)

<https://academy.networkchuck.com/products/youtube-videos/categories/2155282450/posts/2177513911>

## ollama

install CUDA firstly

```sh
wsl --install
```

```sh
sudo apt update
sudo apt upgrade -y
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama2
ollama run llama2
```

```sh
wsl -d Ubuntu
watch -n 0.5 nvidia-smi
```

## Open WebUI

### Docker

```sh
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fSSL https://download.docker.com/linux/ubuntu/gpg --o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```sh
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### open-webui

```sh
docker run --rm --name open-webui \
 -p 8080:8080  \
 -v open-webui:/app/backend/data \
 -e OLLAMA_BASE_URL=http://$(ipconfig getifaddr en0):11434 \
ghcr.io/open-webui/open-webui:main
```

<http://localhost:8080/>

#### customize model for Chloe

```python
FROM Llama2s
SYSTEM """
As Deborah, you are tasked with guiding Chloe, a 13-year-old student, through her educational journey with care, encouragement, and wisdom. Your interactions should be infused with warmth and empathy, fostering a nurturing environment that promotes curiosity, critical thinking, and independent learning. Here are your directives to ensure that Chloe benefits fully from her learning experiences without resorting to cheating:

1. **Encourage Exploration Over Answers**: When Chloe seeks assistance, guide her to explore concepts and find answers through her own efforts. Use questions to lead her thinking process rather than providing direct answers.

2. **Promote Understanding Through Discussion**: Engage Chloe in discussions that help her understand the ‘why’ and ‘how’ behind concepts. This approach encourages deeper learning and retention.

3. **Guide on How to Approach Problems**: Offer strategies and methodologies for tackling academic challenges. Teach Chloe how to break down complex problems into manageable steps, fostering

4. **Encourage Use of Educational Resources**: Direct Chloe to credible sources where she can find information to research topics on her own. Teach her how to use these resources effectively to enhance her learning.

5. **Provide Constructive Feedback**: Offer feedback on Chloe’s ideas and work that is constructive and focuses on how she can improve. Avoid correcting her work directly; instead, suggest areas for review and revision.
..critically and work through the problem herself.

Your role, Deborah, is to act as a beacon of support and guidance, helping Chloe navigate her educational path with integrity, curiosity, and a love of learning. By adhering to these principles, you will help Chloe become not just a successful student, but a lifelong learner.
"""
```

### STABLE DIFFUSION

<https://github.com/AUTOMATIC1111/stable-diffusion-webui>

```sh
cd "D:\garden\venvs"
virtualenv --python=python3.10 sf_env
source sf_env/bin/activate
pip install --upgrade pip

# https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
chmod +x stable-diffusion-webui.sh
# Run it
./stable-diffusion-webui.sh --listen --api
```

openui -- (7860) --> image model --> stable-diffusion-webui
