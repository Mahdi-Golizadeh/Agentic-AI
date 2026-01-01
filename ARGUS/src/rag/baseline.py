import torch
from src.llm.model import load_llm
from src.utils.prompts import BASELINE_RAG_PROMPT


class BaselineRAG:
    def __init__(self, vectordb, k=4):
        self.vectordb = vectordb
        self.k = k
        self.tokenizer, self.model = load_llm()

    def retrieve(self, query):
        docs = self.vectordb.similarity_search(query, k=self.k)
        return docs

    def generate(self, query, docs):
        context = "\n\n".join(
            [f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs]
        )

        prompt = BASELINE_RAG_PROMPT.format(
            context=context,
            question=query
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer(self, query):
        docs = self.retrieve(query)
        return self.generate(query, docs)
