from ollama import Client

from PIL import Image
import base64
import io


def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64


image_path = "D:\\download\\test.jpg"
base64_image = image_to_base64(image_path)

client = Client(host='http://localhost:11434')
response = client.chat(
    model="x/llama3.2-vision:latest",
    messages=[{
        "role": "user",
        "content": "Describe this image?",
        "images": [base64_image]
    }],
)

print(response['message']['content'])
