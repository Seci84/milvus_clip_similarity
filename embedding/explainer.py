import os
import base64
import matplotlib.pyplot as plt
from PIL import Image
from openai import OpenAI


class GPT4oExplainer:
    """
    GPT-4o-based multimodal explanation class.
    - Explain image retrieval results for text queries
    - Explain image-to-image similarity
    - Explain text label relevance for image-to-text search
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.client = OpenAI()
        self.model_name = model_name

    # ---------- Utility ----------
    def encode_image_to_base64(self, path: str) -> str | None:
        """Convert an image file to a base64-encoded string, safely"""
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                
                if len(encoded) < 100:
                    return None
                return encoded
        except Exception as e:
            print(f" Base64 encoding failed for {path}: {e}")
            return None

    def normalize_hits(self, results):
        """
        Normalize the structure of Milvus search results to a consistent list of dicts.
        Supports:
        - [ [ {...}, {...} ] ]
        - [ {...}, {...} ]
        - {"entity": {...}} single dict
        """
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            return results[0]
        elif isinstance(results, list):
            return results
        elif isinstance(results, dict):
            return [results]
        return []

    def extract_label(self, hit):
        if not isinstance(hit, dict):
            return "N/A"
        if "entity" in hit and isinstance(hit["entity"], dict):
            return hit["entity"].get("label", "N/A")
        return hit.get("label", "N/A")

    # def extract_path(self, hit):
    #     if not isinstance(hit, dict):
    #         return ""
    #     if "entity" in hit and isinstance(hit["entity"], dict):
    #         return hit["entity"].get("filepath", "")
    #     return hit.get("filepath", "")

    def extract_path(self, hit):
        if not isinstance(hit, dict):
            return ""
        path = ""
        if "entity" in hit and isinstance(hit["entity"], dict):
            path = hit["entity"].get("filepath", "")
        else:
            path = hit.get("filepath", "")
    
        if path and not os.path.exists(path):
            alt_path = os.path.join("./sample_data", os.path.basename(path))
            if os.path.exists(alt_path):
                path = alt_path
        return path

    
    def extract_score(self, hit):
        if not isinstance(hit, dict):
            return 0
        return hit.get("distance", hit.get("score", 0))    
        

    def print_and_visualize(self, explanation: str, results, top_k: int = 3):
        """Print GPT-4o explanation and visualize top-K retrieved images"""
        print("\n=== GPT-4o Explanation ===\n")
        print(explanation)
        print(f"\n=== Retrieved Images (Top {top_k}) ===\n")

        hits = self.normalize_hits(results)
        plt.figure(figsize=(12, 3))
        for i, hit in enumerate(hits[:top_k]):
            path = self.extract_path(hit)
            label = self.extract_label(hit)
            score = self.extract_score(hit)
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    plt.subplot(1, top_k, i + 1)
                    plt.imshow(img)
                    plt.axis("off")
                    plt.title(f"{label}\n(score={score:.4f})", fontsize=9)
                except Exception as e:
                    print(f" Failed to load image: {path} ({e})")
            else:
                print(f" File not found: {path}")
        plt.show()

    
    # ---------- Text → Image ----------
    def explain_with_text_query(self, query: str, results):
        """Explain retrieved image results for a given text query."""
        hits = self.normalize_hits(results)

        retrieved_context = "\n".join([
            f"Label: {self.extract_label(hit)}, Path: {self.extract_path(hit)}"
            for hit in hits[:3]
        ])

        image_messages = []
        for hit in hits[:3]:
            path = self.extract_path(hit)
            encoded = self.encode_image_to_base64(path)
            if not encoded:
                continue
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            })

        messages = [
            {"role": "system", "content": "You are a visual reasoning assistant for multimodal retrieval-augmented generation."},
            {"role": "user", "content": [
                {"type": "text", "text": f"User query: '{query}'"},
                {"type": "text", "text": f"Retrieved context:\n{retrieved_context}"},
                {"type": "text", "text": "Explain why these retrieved images are relevant to the query."},
                *image_messages
            ]}
        ]

        res = self.client.chat.completions.create(model=self.model_name, messages=messages)
        explanation = res.choices[0].message.content

        self.print_and_visualize(explanation, results)
        return explanation

    # ---------- Image → Image ----------
    def explain_with_image_query(self, image_path: str, results):
        """Explain visual similarities between the query image and retrieved images."""
        hits = self.normalize_hits(results)

        retrieved_context = "\n".join([
            f"Label: {self.extract_label(hit)}, Path: {self.extract_path(hit)}"
            for hit in hits[:3]
        ])

        image_messages = []
        for hit in hits[:3]:
            path = self.extract_path(hit)
            encoded = self.encode_image_to_base64(path)
            if not encoded:
                continue
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            })

        query_image_encoded = self.encode_image_to_base64(image_path)
        if not query_image_encoded:
            raise FileNotFoundError(f" Query image not found or invalid: {image_path}")

        messages = [
            {"role": "system", "content": "You are a visual reasoning assistant that explains image similarities."},
            {"role": "user", "content": [
                {"type": "text", "text": "Given the query image and similar retrieved images, explain their similarities."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image_encoded}"}},
                {"type": "text", "text": f"Retrieved image metadata:\n{retrieved_context}"},
                *image_messages
            ]}
        ]

        res = self.client.chat.completions.create(model=self.model_name, messages=messages)
        explanation = res.choices[0].message.content

        self.print_and_visualize(explanation, results)
        return explanation

    # ---------- Image → Text ----------
    def explain_image_to_text(self, image_path: str, results):
        """Explain why retrieved text labels are relevant to the given image."""
        hits = self.normalize_hits(results)

        labels_context = "\n".join([
            f"{i + 1}. {self.extract_label(hit)} (score={self.extract_score(hit):.4f})"
            for i, hit in enumerate(hits[:5])
        ])

        query_image_encoded = self.encode_image_to_base64(image_path)
        if not query_image_encoded:
            raise FileNotFoundError(f" Query image not found or invalid: {image_path}")

        messages = [
            {"role": "system", "content": "You are a visual reasoning assistant that explains cross-modal retrieval results."},
            {"role": "user", "content": [
                {"type": "text", "text": "Given this image, explain why the following text labels are relevant."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image_encoded}"}},
                {"type": "text", "text": f"Retrieved labels:\n{labels_context}"}
            ]}
        ]

        res = self.client.chat.completions.create(model=self.model_name, messages=messages)
        explanation = res.choices[0].message.content

        print("\n=== GPT-4o Reasoning (Image → Text) ===\n")
        print(explanation)
        return explanation


