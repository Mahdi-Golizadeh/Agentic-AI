import torch
from src.llm.model import load_llm

ANSWER_PROMPT = """<s>[INST]
You answer the QUESTION using ONLY the provided CONTEXT.

Rules:
- Use only information from CONTEXT
- Do NOT add external knowledge
- If the answer is not fully supported, say "The provided documents do not contain enough information."
- Be concise and technical

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
[/INST]"""

class AnswerGenerator:
    def __init__(self):
        self.tokenizer, self.model = load_llm()

    def generate(self, question, documents):
        context = "\n\n".join(
            f"- {doc.page_content}" for doc in documents
        )

        prompt = ANSWER_PROMPT.format(
            question=question,
            context=context
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return answer.strip()
