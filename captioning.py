import os
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch


class FlorenceCaptioner:
    def __init__(self, model_id: str = "microsoft/Florence-2-base"):
        self.model_id = model_id
        self.device = self._get_device()
        self.processor = None
        self.model = None

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self):
        print(f"Using device: {self.device}")
        print(f"Loading {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
        print(f"✅ Model loaded successfully on [{self.device}]")

    def generate_captions(self, image_folder: str, output_csv: str, prompt: str = "<DETAILED_CAPTION>"):
        image_paths = list(Path(image_folder).rglob("*.JPEG"))
        print(f"{len(image_paths)} images → start captioning")
        results = []

        for path in tqdm(image_paths, desc="Captioning"):
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=64)
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results.append({"image_path": str(path), "caption": caption})
            except Exception as e:
                results.append({"image_path": str(path), "caption": f"ERROR: {e}"})

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ Completed! {len(df)} captions saved → {output_csv}")



   


