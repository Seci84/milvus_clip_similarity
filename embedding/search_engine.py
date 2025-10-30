import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient


class CLIPSearchEngine:

    def __init__(self, 
                 collection_name: str,
                 model_name: str = "clip-ViT-B-32",
                 milvus_uri: str = "http://localhost:19530",
                 text_field: str = "vector_clip_text",
                 image_field: str = "vector_clip_image"):
        """
        Args:
            collection_name (str): Name of the Milvus collection.
            model_name (str): Name of the SentenceTransformer CLIP model.
            milvus_uri (str): URI address of the Milvus server.
            text_field (str): Field name for text embeddings in Milvus.
            image_field (str): Field name for image embeddings in Milvus.
        """
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.client = MilvusClient(uri=milvus_uri)
        self.text_field = text_field
        self.image_field = image_field

    # embedding
    def encode_text(self, text: str):
        return self.model.encode(text).tolist()

    def encode_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        return self.model.encode(img).tolist()

    # search
    def search(self, query_vec, anns_field: str, limit: int = 5):
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            anns_field=anns_field,
            limit=limit,
            output_fields=["label", "filepath"]
        )
        return [
            {
                "label": hit["entity"]["label"],
                "filepath": hit["entity"]["filepath"],
                "score": hit["distance"]
            }
            for hit in res[0]
        ]

    # text → image
    def search_by_text(self, query: str, limit: int = 5):
        query_vec = self.encode_text(query)
        return self.search(query_vec, self.text_field, limit)

    # image → image
    def search_by_image(self, image_path: str, limit: int = 5):
        query_vec = self.encode_image(image_path)
        return self.search(query_vec, self.image_field, limit)

    # image → text
    def search_text_by_image(self, image_path: str, limit: int = 5):
        query_vec = self.encode_image(image_path)
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            anns_field=self.image_field,
            limit=limit,
            output_fields=["label", "filepath"]
        )

        print("\n Image → Text results:")
        for i, hit in enumerate(res[0], 1):
            print(f"{i}. {hit['entity']['label']} (score={hit['distance']:.4f})")

        return [
            {
                "rank": i + 1,
                "label": hit["entity"]["label"],
                "filepath": hit["entity"]["filepath"],
                "score": hit["distance"]
            }
            for i, hit in enumerate(res[0])
        ]
