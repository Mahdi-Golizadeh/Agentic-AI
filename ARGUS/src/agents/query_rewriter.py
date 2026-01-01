import torch
from src.llm.model import load_llm

REWRITE_PROMPT = """<s>[INST]
You rewrite search queries for vector retrieval.

Rules:
- Output ONE single sentence
- Do NOT enumerate
- Do NOT ask multiple questions
- Be concise and specific
- No explanations

Rewrite the query to improve document retrieval.

Original query:
{query}

Rewritten query:
[/INST]"""

class QueryRewriter:
    def __init__(self):
        self.tokenizer, self.model = load_llm()

    def rewrite(self, query: str) -> str:
        prompt = REWRITE_PROMPT.format(query=query)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # ✅ Extract only NEW tokens (critical)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # ✅ Enforce single-line retrieval query
        response = response.strip().split("\n")[0]

        return response
