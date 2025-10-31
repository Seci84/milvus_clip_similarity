# Multimodal Retrieval & Reasoning System  
**Image–Text Search and Interpretation using CLIP + Milvus + GPT-4o**

---

## Overview
This repository implements a **multimodal retrieval and reasoning system** that connects **visual and textual semantics**.  
It combines:

- **CLIP** — for joint image–text embeddings  
- **Milvus** — for scalable vector similarity search  
- **GPT-4o** — for natural-language reasoning and explanation  

The system retrieves the most semantically relevant images or captions for a given query  
and generates an explanation describing *why* they are related.

---

##  System Architecture
CLIP Encoder -> Milvus Vector Database -> Re-Ranking Layer -> GPT-4o Explainer 

---

## ⚙️ Core Components

### 1. `CLIPSearchEngine`
A unified interface for multimodal embedding and retrieval using CLIP and Milvus.

**Main Functions**
- `encode_text(text)` — Encode text into CLIP embeddings  
- `encode_image(path)` — Encode images into CLIP embeddings  
- `search_by_text()` — Retrieve visually similar images for a text query  
- `search_by_image()` — Retrieve similar images for an image query  
- `search_text_by_image()` — Retrieve related textual labels for an image  

**Dependencies**
- [`SentenceTransformer("clip-ViT-B-32")`](https://huggingface.co/sentence-transformers/clip-ViT-B-32)  
- [`pymilvus`](https://milvus.io/docs/install_standalone-docker.md)

---

### 2. `rerank_results()`
Combines image-based and text-based search results by weighted score fusion.

```python
score_final = α * score_img + (1 - α) * score_txt


### 3. GPT4oExplainer
Provides semantic reasoning for retrieved results using OpenAI’s GPT-4o.

[Functions]
explain_with_text_query() — Explain why images match a text query
explain_with_image_query() — Explain similarities between retrieved images
explain_image_to_text() — Explain why certain labels fit a given image

[How it Works]
Encode top-N retrieved images into Base64
Send both images and text metadata to GPT-4o
Receive a natural-language explanation describing visual–semantic alignment

## Data Flow
| Stage             | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| **1. Embedding**  | CLIP encodes text and images into a shared latent space     |
| **2. Storage**    | Embeddings are stored in Milvus                             |
| **3. Retrieval**  | Query (text or image) is encoded and matched via ANN search |
| **4. Re-Ranking** | Weighted combination of image and text similarity scores    |
| **5. Reasoning**  | GPT-4o interprets and explains the semantic relationship    |

## Tech Stack
| Category               | Tool                                   | Role                                        |
| ---------------------- | -------------------------------------- | ------------------------------------------- |
| **Multimodal Encoder** | `SentenceTransformer("clip-ViT-B-32")` | CLIP-based text/image embedding             |
| **Vector Database**    | `Milvus`                               | Stores embeddings, performs ANN search      |
| **Re-Ranking**         | Python weighted fusion (`α`)           | Combines scores from both modalities        |
| **Reasoning Engine**   | `GPT-4o`                               | Explains results using multimodal reasoning |
| **Visualization**      | `matplotlib`, `Pillow`                 | Image preview and comparison                |


## Example Output
=== GPT-4o Explanation ===

The query image shows a bowl of dough.
Here are some similarities with the retrieved images:
1. All images feature dough in a mixing bowl.
2. The dough color and texture are similar (soft beige tone).
3. Each image centers compositionally on the bowl itself.

