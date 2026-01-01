import json
import torch
from src.llm.model import load_llm

GRADER_PROMPT = GRADER_PROMPT = """You are a binary relevance classifier.

Decide whether the DOCUMENT CHUNK helps answer the QUESTION.

Respond ONLY with a valid JSON object in this exact format:
{{"relevant":"yes","reason":"short reason"}}

QUESTION:
{question}

DOCUMENT CHUNK:
{context}

JSON:
"""

class RelevanceGrader:
    def __init__(self):
        self.tokenizer, self.model = load_llm()

    def grade(self, question, document):
        prompt = GRADER_PROMPT.format(
            question=question,
            context=document.page_content
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print("RAW MODEL OUTPUT:\n", response)
        return self._parse_response(response)

    def _parse_response(self, response):
        try:
            json_str = response[response.index("{"):response.rindex("}") + 1]
            parsed = json.loads(json_str)

            if parsed.get("relevant") not in ["yes", "no"]:
                raise ValueError("Invalid relevance value")

            return parsed

        except Exception:
            return {
                "relevant": "no",
                "reason": "Model did not return valid JSON"
            }

