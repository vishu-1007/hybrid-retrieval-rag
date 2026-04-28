"""
generate_kb.py
Generates synthetic support ticket knowledge base entries using any LLM via litellm.

Usage:
    python generate_kb.py --count 10
    python generate_kb.py --categories Payment Order --count 5
    python generate_kb.py --model claude-3-haiku-20240307 --count 10
    python generate_kb.py --count 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import pandas as pd

try:
    import litellm
except ImportError:
    print("litellm not installed. Run: pip install litellm")
    sys.exit(1)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_KB_PATH = "data/knowledge_base.csv"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_RETRIES = 3

ALL_CATEGORIES = [
    "Payment", "Account", "Order", "Delivery",
    "Technical", "Coupon", "Subscription", "Wallet", "Fraud",
]

SCHEMA_DESCRIPTION = """
Return a JSON array of objects. Each object MUST have exactly these keys:
- category        : one of the given categories
- subcategory     : a specific sub-type of the issue (2-4 words)
- journey_stage   : customer journey stage (e.g. Onboarding, Checkout, Post-Purchase, Delivery, App Usage, Account Management, Billing, Account Security)
- issue_description: 1-2 sentences describing the customer complaint
- resolution_steps : a single string with numbered steps separated by '. '

Return ONLY the JSON array with no extra text.
"""


def generate_batch(
    categories: list[str],
    count: int,
    model: str,
    api_base: str | None,
    temperature: float,
    max_retries: int,
) -> list[dict]:
    targets = random.choices(categories, k=count)
    category_list = ", ".join(f'"{c}"' for c in targets)

    prompt = f"""Generate {count} unique customer support knowledge base entries for an e-commerce platform.
Target categories (one per entry, in order): [{category_list}]

{SCHEMA_DESCRIPTION}
"""

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if api_base:
        kwargs["api_base"] = api_base

    for attempt in range(1, max_retries + 1):
        try:
            response = litellm.completion(**kwargs)
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            entries = json.loads(raw)
            if isinstance(entries, list):
                return entries
        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(2 ** attempt)

    return []


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic KB entries")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--kb-path", default=DEFAULT_KB_PATH)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    categories = args.categories or ALL_CATEGORIES
    invalid = [c for c in categories if c not in ALL_CATEGORIES]
    if invalid:
        print(f"Unknown categories: {invalid}. Valid: {ALL_CATEGORIES}")
        sys.exit(1)

    print(f"Generating {args.count} entries using model: {args.model}")
    all_entries: list[dict] = []
    remaining = args.count

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        print(f"  Requesting batch of {batch_size}...")
        batch = generate_batch(
            categories, batch_size, args.model,
            args.api_base, args.temperature, args.max_retries
        )
        all_entries.extend(batch)
        remaining -= len(batch)
        if not batch:
            print("  Empty batch returned — stopping.")
            break

    if not all_entries:
        print("No entries generated.")
        sys.exit(1)

    df_new = pd.DataFrame(all_entries)
    # Ensure required columns exist
    required = ["category", "subcategory", "journey_stage", "issue_description", "resolution_steps"]
    for col in required:
        if col not in df_new.columns:
            df_new[col] = ""
    df_new = df_new[required]

    if args.dry_run:
        print("\n--- Generated entries (dry run) ---")
        print(df_new.to_string(index=False))
        return

    output_path = args.output or args.kb_path
    if os.path.exists(output_path) and output_path == args.kb_path:
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_path, index=False)
        print(f"✅ Appended {len(df_new)} entries to {output_path} (total: {len(df_combined)})")
    else:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        df_new.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df_new)} entries to {output_path}")


if __name__ == "__main__":
    main()
