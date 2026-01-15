import pandas as pd
from pathlib import Path
import sys
from utils.data_handler import load_csv
from utils.arabic_text import (
    remove_arabic_noise,
    remove_stopwords,
    normalize_arabic,
    stem_text,
    lemmatize_text
)


def preprocess_main(args):
    if args.preprocess_command == 'remove':
        remove_cmd(args)
    elif args.preprocess_command == 'stopwords':
        stopwords_cmd(args)
    elif args.preprocess_command == 'replace':
        replace_cmd(args)
    elif args.preprocess_command == 'all':
        all_cmd(args)
    elif args.preprocess_command == 'stem':
        stem_cmd(args)
    elif args.preprocess_command == 'lemmatize':
        lemmatize_cmd(args)
    else:
        print("Error: No Preprocess subcommand specified")
        print("Available subcommands: remove, stopwords, replace, stem, lemmarize, all")
        sys.exit(1)



def _load_and_check(csv_path, text_col):
    df = load_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV")
    return df


def remove_cmd(args):
    print(f"Removing noise from {args.csv_path}...")
    
    df = _load_and_check(args.csv_path, args.text_col)
    
    before = df[args.text_col].astype(str).str.len()
    df[args.text_col] = df[args.text_col].apply(
        lambda x: remove_arabic_noise(str(x), args.remove)
    )
    after = df[args.text_col].astype(str).str.len()

    _save(df, args.output)

    _print_stats("chars", before, after, args.output)


def stopwords_cmd(args):
    print(f"Removing stopwords from {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)

    before = df[args.text_col].astype(str).str.split().str.len()
    df[args.text_col] = df[args.text_col].apply(
        lambda x: remove_stopwords(str(x), args.sw_path, args.language)
    )
    after = df[args.text_col].astype(str).str.split().str.len()

    _save(df, args.output)
    _print_stats("words", before, after, args.output)


def replace_cmd(args):
    print(f"Normalizing Arabic text in {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)
    df[args.text_col] = df[args.text_col].apply(
        lambda x: normalize_arabic(str(x))
    )

    _save(df, args.output)
    print(f"Normalized data saved to: {args.output}")


def all_cmd(args):
    print(f"Running full preprocessing pipeline on {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)
    before = df[args.text_col].astype(str).str.len()

    print("  [1/4] Removing noise...")
    df[args.text_col] = df[args.text_col].apply(
        lambda x: remove_arabic_noise(str(x), 'all')
    )

    print("  [2/4] Removing stopwords...")
    df[args.text_col] = df[args.text_col].apply(
        lambda x: remove_stopwords(str(x), args.sw_path, args.language)
    )

    print("  [3/4] Normalizing text...")
    df[args.text_col] = df[args.text_col].apply(
        lambda x: normalize_arabic(str(x))
    )
    
    print("  [4/4] stemming text...")
    df[args.text_col] = df[args.text_col].apply(
        lambda x: stem_text(str(x), args.language, args.stemmer)
    )

    after = df[args.text_col].astype(str).str.len()
    _save(df, args.output)
    _print_stats("chars", before, after, args.output)


def stem_cmd(args):
    print(f"Stemming text in {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)
    df[args.text_col] = df[args.text_col].apply(
        lambda x: stem_text(str(x), args.language, args.stemmer)
    )

    _save(df, args.output)
    print(f"Stemmed data saved to: {args.output}")


def lemmatize_cmd(args):
    print(f"Lemmatizing text in {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)
    df[args.text_col] = df[args.text_col].apply(
        lambda x: lemmatize_text(str(x), args.language)
    )

    _save(df, args.output)
    print(f"Lemmatized data saved to: {args.output}")



def _save(df, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def _print_stats(unit, before, after, output):
    print("\nStatistics:")
    print(f"  Count {unit} before: {sum(before)}")
    print(f"  Count {unit} after:  {sum(after)}")
    print(f"  Reduction: {((before.mean() - after.mean()) / before.mean() * 100):.2f}%")
    print(f"\nSaved to: {output}")
