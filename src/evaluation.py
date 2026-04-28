"""
src/evaluation.py
RAG Evaluation using new google-genai SDK.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from google import genai

EVAL_MODEL = "models/gemini-2.5-flash"  # ✅ FIXED

TEST_CASES = [
    {"complaint": "I paid via UPI but my order was never confirmed. Amount got debited.", "expected_category": "Payment", "expected_subcategory": "Failed Transaction"},
    {"complaint": "I'm not receiving the OTP on my phone to complete payment.", "expected_category": "Payment", "expected_subcategory": "OTP Issue"},
    {"complaint": "My account got locked after too many wrong password attempts.", "expected_category": "Account", "expected_subcategory": "Account Locked"},
    {"complaint": "The app crashes every time I try to open my cart.", "expected_category": "Technical", "expected_subcategory": "App Crash"},
    {"complaint": "I received a completely damaged product. The box was crushed.", "expected_category": "Delivery", "expected_subcategory": "Damaged Package"},
    {"complaint": "My refund hasn't been credited even after 10 days.", "expected_category": "Payment", "expected_subcategory": "Refund Delay"},
    {"complaint": "The discount coupon I have is not getting applied at checkout.", "expected_category": "Coupon", "expected_subcategory": "Code Not Applied"},
    {"complaint": "There's a suspicious transaction on my account I never made.", "expected_category": "Fraud", "expected_subcategory": "Unauthorized Transaction"},
    {"complaint": "I subscribed to the premium plan but it's still showing free.", "expected_category": "Subscription", "expected_subcategory": "Not Activated"},
    {"complaint": "My wallet balance didn't update after I recharged it.", "expected_category": "Wallet", "expected_subcategory": "Balance Not Updated"},
]


@dataclass
class RetrievalResult:
    complaint: str
    expected_category: str
    expected_subcategory: str
    retrieved_categories: list
    retrieved_subcategories: list
    hybrid_scores: list
    hit: bool
    sub_hit: bool
    reciprocal_rank: float
    avg_hybrid_score: float


@dataclass
class GenerationResult:
    complaint: str
    llm_category: str
    llm_subcategory: str
    faithfulness: float
    answer_relevance: float
    faithfulness_reason: str
    relevance_reason: str


class RAGEvaluator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def eval_retrieval(self, retriever, top_k=3, alpha=0.5):
        results = []
        for tc in TEST_CASES:
            retrieved = retriever.retrieve(tc["complaint"], top_k=top_k, alpha=alpha)

            cats = [r["category"] for r in retrieved]
            subcats = [r["subcategory"] for r in retrieved]
            scores = [r["hybrid_score"] for r in retrieved]

            hit = tc["expected_category"] in cats
            sub_hit = tc["expected_subcategory"] in subcats

            rr = 0.0
            for rank, cat in enumerate(cats, 1):
                if cat == tc["expected_category"]:
                    rr = 1.0 / rank
                    break

            results.append(RetrievalResult(
                complaint=tc["complaint"],
                expected_category=tc["expected_category"],
                expected_subcategory=tc["expected_subcategory"],
                retrieved_categories=cats,
                retrieved_subcategories=subcats,
                hybrid_scores=scores,
                hit=hit,
                sub_hit=sub_hit,
                reciprocal_rank=rr,
                avg_hybrid_score=sum(scores) / len(scores) if scores else 0.0,
            ))

        return results

    def _judge(self, complaint, context, response):
        prompt = f"""You are an expert RAG evaluator.

CUSTOMER COMPLAINT:
{complaint}

RETRIEVED CONTEXT:
{context}

LLM RESPONSE:
{response}

Score strictly:

- FAITHFULNESS (0-1): Are answers grounded in context?
- ANSWER_RELEVANCE (0-1): Does it solve the complaint?

Return ONLY JSON:
{{"faithfulness": 0.0, "faithfulness_reason": "reason",
"answer_relevance": 0.0, "relevance_reason": "reason"}}"""

        resp = self.client.models.generate_content(
            model=EVAL_MODEL,
            contents=prompt,
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )

        raw = resp.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "faithfulness": 0.0,
                "faithfulness_reason": "Parse error",
                "answer_relevance": 0.0,
                "relevance_reason": "Parse error",
            }

    def eval_generation(self, retriever, generator, top_k=3, alpha=0.5):
        results = []

        for tc in TEST_CASES:
            retrieved = retriever.retrieve(tc["complaint"], top_k=top_k, alpha=alpha)
            llm_output = generator.generate(tc["complaint"], retrieved)

            context_str = "\n".join(
                f"[{r['category']} > {r['subcategory']}]: {r['issue_description']} | {r['resolution_steps']}"
                for r in retrieved
            )

            try:
                scores = self._judge(
                    tc["complaint"],
                    context_str,
                    json.dumps(llm_output)
                )
                time.sleep(0.3)  # ✅ slight rate limit protection
            except Exception:
                scores = {
                    "faithfulness": 0.0,
                    "faithfulness_reason": "Error",
                    "answer_relevance": 0.0,
                    "relevance_reason": "Error",
                }

            results.append(GenerationResult(
                complaint=tc["complaint"],
                llm_category=llm_output.get("category", ""),
                llm_subcategory=llm_output.get("subcategory", ""),
                faithfulness=scores.get("faithfulness", 0.0),
                answer_relevance=scores.get("answer_relevance", 0.0),
                faithfulness_reason=scores.get("faithfulness_reason", ""),
                relevance_reason=scores.get("relevance_reason", ""),
            ))

        return results