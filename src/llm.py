"""
src/llm.py
LLM generation using Google GenAI SDK (Gemini 2.5 Flash).
"""

from __future__ import annotations

import json
from google import genai

LLM_MODEL = "models/gemini-2.5-flash"

SYSTEM_PROMPT = """You are an expert customer support AI assistant.
Given a customer complaint and relevant knowledge base context,
classify the complaint and provide actionable resolution steps.

You MUST respond with ONLY a valid JSON object in exactly this schema:
{
  "category": "string",
  "subcategory": "string",
  "journey_stage": "string",
  "confidence": "High | Medium | Low",
  "summary": "one-sentence summary of the complaint",
  "resolution_steps": ["step 1", "step 2", "step 3"]
}

Do not include any text outside the JSON object. No markdown, no backticks.
"""


def build_prompt(complaint: str, context_entries: list[dict]) -> str:
    context_text = ""
    for i, entry in enumerate(context_entries, 1):
        context_text += (
            f"\n[{i}] Category: {entry['category']} > {entry['subcategory']}\n"
            f"    Stage: {entry['journey_stage']}\n"
            f"    Issue: {entry['issue_description']}\n"
            f"    Resolution: {entry['resolution_steps']}\n"
        )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Customer Complaint:\n\"\"\"{complaint}\"\"\"\n\n"
        f"Relevant Knowledge Base Context:{context_text}\n\n"
        f"Respond with the JSON object only."
    )


class LLMGenerator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, complaint: str, context_entries: list[dict]) -> dict:
        prompt = build_prompt(complaint, context_entries)

        response = self.client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )

        raw = response.text.strip()

        # Fallback cleanup (rare but safe)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "category": "Unknown",
                "subcategory": "Unknown",
                "journey_stage": "Unknown",
                "confidence": "Low",
                "summary": "Could not parse response.",
                "resolution_steps": [raw],
            }