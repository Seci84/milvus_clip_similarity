from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableMap,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
import re
import os
from IPython.display import Image, display


class FlorenceRAG:
    """
    Florence Caption 기반 이미지 검색 + LLM Reasoning RAG 클래스
    ----------------------------------------------------------
    1. SentenceTransformer 임베딩
    2. Milvus 벡터 검색
    3. LangChain 기반 Stuff Chain + LLM 응답
    """

    def __init__(self, collection_name="florence_clip_miniLM", milvus_uri="http://localhost:19530", model_name="gpt-4o"):
        # Milvus 및 LLM 구성
        self.collection_name = collection_name
        self.client = MilvusClient(uri=milvus_uri)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # Prompt 템플릿 설정
        template = """
        You are a visual reasoning assistant with access to image captions and filepaths.

        User question:
        {question}

        Below are the top related image captions and filepaths:
        {context}

        Instructions:
        - If the question asks for an image, copy and return the exact Filepath as shown in the context (do not modify or paraphrase it).
        - If multiple relevant images exist, pick one representative filepath.
        - If counting is required, count accurately.

        Answer:
        """
        self.prompt = PromptTemplate.from_template(template)

        # 체인 생성
        self.qa_chain = self._create_stuff_chain(self.llm, self.prompt)

    # Embedding 함수

    def embed_texts(self, texts):
        """텍스트를 벡터로 변환"""
        return self.embedder.encode(texts).tolist()


    # Milvus 검색 함수

    def get_relevant_documents(self, query: str, limit=30): # limit=None으로 둘 경우 토큰 한도 초과 가능성 고려
        """Milvus에서 유사한 caption 검색"""
        qv = self.embed_texts([query])
        res = self.client.search(
            collection_name=self.collection_name,
            data=qv,
            anns_field="vector_clip_text",
            limit=limit,
            search_params={"metric_type": "IP", "params": {"nprobe": 10}},
            output_fields=["caption", "filepath", "metadata"]
        )

        docs = []
        for hit in res[0]:
            caption = hit["entity"].get("caption", "")
            filepath = hit["entity"].get("filepath", "")
            docs.append(f"Caption: {caption}\nFilepath: {filepath}")
        return docs


    # Stuff Chain 구성

    def _create_stuff_chain(self, llm, prompt):
        """문서를 합쳐 Prompt에 삽입하는 체인"""
        def combine_docs(inputs):
            docs = inputs.get("context", [])
            if not docs:
                return ""
            if isinstance(docs[0], str):
                return "\n\n".join(docs)
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            RunnableLambda(lambda inputs: {
                "context": combine_docs(inputs),
                "question": inputs.get("question", "")
            })
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain


    # RAG Chain 통합

    def build_rag_chain(self):
        """Retriever + QA Chain 통합"""
        return (
            RunnableMap({
                "context": RunnableLambda(lambda x: self.get_relevant_documents(x)),
                "question": RunnablePassthrough()
            })
            | RunnableLambda(lambda inputs: {
                "context": inputs["context"],
                "question": inputs["question"]
            })
            | self.qa_chain
            | StrOutputParser()
        )


    # 결과 시각화

    def show_image_from_response(self, response):
        """LLM 응답 내 파일 경로를 탐색하고 이미지 표시"""
        match = re.search(r"([^\s]+\.jpe?g|[^\s]+\.png)", response, re.IGNORECASE)
        if match:
            image_path = match.group(1)
            if os.path.exists(image_path):
                print(f"\n Showing image: {image_path}")
                display(Image(filename=image_path))
            else:
                print(f"\n Image not found at: {image_path}")
        else:
            print("\n No image path detected in LLM response.")


