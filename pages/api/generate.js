import { hf_hub_download } from "@huggingface/hub";
import { FluxKontextPipeline } from "diffusers";
import fetch from "node-fetch";

export default async function handler(req, res) {
  try {
    const { prompt, style } = req.body;
    if (!prompt || !style)
      return res.status(400).json({ error: "Missing prompt or style" });

    // โหลด LoRA weights จาก Hugging Face Hub
    const loraPath = hf_hub_download({
      repo_id: "Owen777/Kontext-Style-Loras",
      filename: `${style}_lora_weights.safetensors`,
    });

    // โหลด pipeline
    const pipeline = await FluxKontextPipeline.from_pretrained(
      "black-forest-labs/FLUX.1-Kontext-dev"
    );
    pipeline.to("cuda");
    pipeline.load_lora_weights(loraPath, { adapter_name: "lora" });
    pipeline.set_adapters(["lora"], { adapter_weights: [1] });

    // โหลด image ตัวอย่าง
    const imageUrl =
      "https://huggingface.co/datasets/black-forest-labs/kontext-bench/resolve/main/test/images/0003.jpg";
    const response = await fetch(imageUrl);
    const buffer = await response.arrayBuffer();
    const image = await pipeline.utils.load_image(Buffer.from(buffer));

    // Generate ภาพ
    const result = await pipeline({
      prompt,
      image,
      height: 512,
      width: 512,
      num_inference_steps: 24,
    });
    const base64Image = await result.images[0].to_base64();

    res.status(200).json({ image: `data:image/png;base64,${base64Image}` });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
}
