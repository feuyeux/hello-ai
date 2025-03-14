# OpenManus

```sh
Install uv (A fast Python package installer and resolver):
curl -LsSf https://astral.sh/uv/install.sh | sh
Clone the repository:
git clone https://github.com/mannaandpoem/OpenManus.git
cd OpenManus
Create a new virtual environment and activate it:
uv venv
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
Install dependencies:
uv pip install -r requirements.txt
```

```sh
cp config/config.example.toml config/config.toml
```

```toml
# Global LLM configuration

[llm]
model = "gpt-4o"
base_url = "<https://api.openai.com/v1>"
api_key = "sk-..."  # Replace with your actual API key
max_tokens = 4096
temperature = 0.0

# Optional configuration for specific LLM models

[llm.vision]
model = "gpt-4o"
base_url = "<https://api.openai.com/v1>"
api_key = "sk-..."  # Replace with your actual API keyv
```

```sh
python main.py
```

Then input your idea via terminal!
