import os
import time
import random
from pathlib import Path
import sys
import pandas as pd


def generate_main(args):
    print(f"Generating {args.count} synthetic Arabic texts for class '{args.class_name}'...")
    print(f"   Model: {args.model}")

    # API key retrieval
    api_key = args.api_key
    if not api_key and args.model in ['gemini']:
        env_var = 'GEMINI_API_KEY' 
        api_key = os.getenv(env_var)

    if not api_key and args.model in ['gemini']:
        raise ValueError(
            f"API key required. Set {args.model.upper()}_API_KEY or use --api_key"
        )

    # ---------------- GENERATION ---------------- #
    if args.model == 'gemini':
        texts = generate_with_gemini(
            api_key, args.class_name, args.count, args.prompt, args.temperature
        )
    else:
        print("Error: model name is unavailable.")
        print("Available models: geminit")
        sys.exit(1)

    if not texts:
        raise RuntimeError("Failed to generate texts")

    # ---------------- BUILD DATAFRAME ---------------- #
    records = []
    for i, text in enumerate(texts):
        records.append({
            'id': f'{args.class_name}_{i+1}',
            'text': text.strip(),
            'class': args.class_name,
            'source': 'synthetic',
            'model': args.model
        })

    df = pd.DataFrame(records)

    # ---------------- SAVE ---------------- #
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(args.append)
    if args.append and output_path.exists():
        print(args.append)
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
        print("   Appending to existing file...")

    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nGeneration Summary:")
    print(f"   Generated samples: {len(texts)}")
    print(f"   Average length: {df['text'].str.len().mean():.0f} characters")
    print(f"   Class: {args.class_name}")
    print(f"\nData saved to: {output_path}")


# ============================================================
# GENERATION BACKENDS
# ============================================================
def generate_with_gemini(api_key, class_name, count, custom_prompt, temperature):
    """
    Generate texts using Google Gemini (google.genai SDK)
    """
    try:
        from google import genai
        from google.genai import types

    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)

    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = f"""Generate {count} diverse Arabic text samples.
Class: {class_name}

Requirements:
- Modern Standard Arabic
- 2–4 sentences each
- One text per line
- No numbering or explanations
"""

    print("   Generating with Gemini API (google.genai)...")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature =  temperature,
            maxOutputTokens= 8192,
        ),
    )

    # Parse output
    raw_text = response.text or ""
    texts = [
        t.strip().lstrip("0123456789.-) ")
        for t in raw_text.split("\n")
        if len(t.strip()) > 20
    ]

    print(f"   ✓ Generated {len(texts)} texts")
    return texts[:count]
