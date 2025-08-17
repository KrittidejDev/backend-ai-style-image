import sys, base64
from io import BytesIO
from PIL import Image
from diffusers import FluxKontextPipeline
import torch

prompt, style, image_b64 = sys.argv[1], sys.argv[2], sys.argv[3]

image_data = base64.b64decode(image_b64.split(",")[1])
image = Image.open(BytesIO(image_data)).convert("RGB").resize((256, 256))

pipeline = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
pipeline.load_lora_weights(f"./LoRAs/{style}_lora_weights.safetensors", adapter_name="lora")
pipeline.set_adapters(["lora"], adapter_weights=[1])

result = pipeline(image=image, prompt=f"{prompt}, {style} style", height=256, width=256, num_inference_steps=24).images[0]

buffer = BytesIO()
result.save(buffer, format="PNG")
print("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
