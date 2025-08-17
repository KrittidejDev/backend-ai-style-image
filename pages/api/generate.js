import { spawn } from "child_process";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();
  const { prompt, style, imageBase64 } = req.body;
  if (!prompt || !style || !imageBase64)
    return res.status(400).json({ error: "Missing data" });

  const py = spawn("python", [
    "./scripts/generate.py",
    prompt,
    style,
    imageBase64,
  ]);

  let output = "";
  let error = "";

  py.stdout.on("data", (data) => {
    output += data.toString();
  });
  py.stderr.on("data", (data) => {
    error += data.toString();
  });

  py.on("close", (code) => {
    if (code !== 0)
      return res.status(500).json({ error: error || "Python script failed" });
    res.status(200).json({ image: output.trim() });
  });
}
