# https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUFhttps://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF

from llama_cpp import Llama

llm = Llama(
    model_path="./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
)

output = llm(
    "Q: Name the planets in the solar system? A: ",  # Prompt
    max_tokens=32,
    # Stop generating just before the model would generate a new question
    stop=["Q:", "\n"],
    echo=True  # Echo the prompt back in the output
)  # Generate a completion, can also call create_completion
print(output)
