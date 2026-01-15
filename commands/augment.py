"""
Dataset Augmentation Command
"""

import os
import time
from pathlib import Path

import pandas as pd

from commands.generate import (
    generate_with_gemini,
    generate_with_openai
)


def augment_main(args):
    print(f"📈 Augmenting dataset from {args.input_csv}...")

    df = pd.read_csv(args.input_csv)

    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError("Required columns not found")

    print(f"   Original samples: {len(df)}")
    print(f"   Classes: {df[args.label_col].nunique()}")

    api_key = args.api_key
    if not api_key:
        env_var = 'GEMINI_API_KEY' if args.model == 'gemini' else 'OPENAI_API_KEY'
        api_key = os.getenv(env_var)

    if not api_key:
        raise ValueError("API key required")

    all_new = []

    for class_name in df[args.label_col].unique():
        print(f"\n   Generating {args.samples_per_class} samples for '{class_name}'")

        examples = df[df[args.label_col] == class_name][args.text_col].head(3).tolist()
        examples_text = '\n'.join(f"- {e}" for e in examples)

        prompt = f"""Generate Arabic texts similar to these:

{examples_text}

Requirements:
- Same topic and style
- Modern Standard Arabic
- One per line
"""

        if args.model == 'gemini':
            texts = generate_with_gemini(
                api_key, class_name, args.samples_per_class, prompt, 0.9
            )
        else:
            texts = generate_with_openai(
                api_key, class_name, args.samples_per_class, prompt, 0.9
            )

        if texts:
            for i, text in enumerate(texts):
                all_new.append({
                    'id': f'aug_{class_name}_{i+1}',
                    'text': text.strip(),
                    'class': class_name,
                    'source': 'augmented'
                })

        time.sleep(1)

    new_df = pd.DataFrame(all_new)
    combined = pd.concat([df, new_df], ignore_index=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\n📊 Augmentation Summary:")
    print(f"   Original samples: {len(df)}")
    print(f"   Generated samples: {len(new_df)}")
    print(f"   Total samples: {len(combined)}")
    print(f"\n✅ Augmented data saved to: {args.output}")
