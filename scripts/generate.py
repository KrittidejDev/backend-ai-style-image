from huggingface_hub import hf_hub_download
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
import torch

# 1. เลือก style
STYLE_NAME = "3D_Chibi"

# 2. โหลดไฟล์ LoRA จาก Hugging Face runtime
lora_path = hf_hub_download(
    repo_id="Owen777/Kontext-Style-Loras",
    filename=f"{STYLE_NAME}_lora_weights.safetensors",
    local_dir="./LoRAs"
)

# 3. โหลด pipeline
pipeline = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to('cuda')

# 4. โหลด LoRA weights
pipeline.load_lora_weights(lora_path, adapter_name="lora")
pipeline.set_adapters(["lora"], adapter_weights=[1])

# 5. โหลดภาพตัวอย่าง (หรือรับจาก frontend)
image = load_image("https://huggingface.co/datasets/black-forest-labs/kontext-bench/resolve/main/test/images/0003.jpg").resize((512, 512))

# 6. Generate ภาพ
result = pipeline(
    image=image,
    prompt=f"Turn this image into the {STYLE_NAME.replace('_', ' ')} style.",
    height=512,
    width=512,
    num_inference_steps=24
).images[0]

# 7. บันทึกภาพ
result.save(f"{STYLE_NAME}_generated.png")
