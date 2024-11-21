import os
from autogen import ConversableAgent

local_llm_config = {
    "config_list": [
        {
            "model": "NotRequired",  # Loaded with LiteLLM command
            "api_key": "NotRequired",  # Not needed
            "base_url": "http://0.0.0.0:4000"  # Your LiteLLM URL
        }
    ],
    "cache_seed": None  # Turns off caching, useful for testing different models
}

# read MOONSHOT_API_KEY from environment variable
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

kimi_llm_config = {
    "config_list": [
        {
            "model": "moonshot-v1-8k",
            "api_key":  MOONSHOT_API_KEY,
            "base_url": "https://api.moonshot.cn/v1/"
        }],
    "cache_seed": None
}

cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
    llm_config=kimi_llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
)
joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config=local_llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
)
result = joe.initiate_chat(
    cathy, message="Cathy, 来个段子，要高级，不许谐音梗哦", max_turns=2)
print(result)
