import sys
import argparse
import pandas as pd
from pathlib import Path
from utils.data_handler import load_csv
from utils.visualization import plot_distribution, plot_histogram, generate_wordcloud
from utils.visualization import plot_top_words_per_class
from utils.visualization import plot_ngrams_per_class
from utils.arabic_text import get_text_stats

def distribution(args):
    """
    Visualize class distribution in the dataset
    
    Example:
        python main.py eda distribution --csv_path data.csv --label_col class
    """
    print(f"Analyzing class distribution in {args.csv_path}...")
    
    # Load data
    df = load_csv(args.csv_path)
    
    if args.label_col not in df.columns:
        print(f"Error: Column '{args.label_col}' not found in CSV", file=sys.stderr)
        return
    
    # Generate visualization
    output_path = plot_distribution(df, args.label_col, args.plot_type, args.output)
    
    # Print statistics
    counts = df[args.label_col].value_counts()
    print("\n Class Distribution:")
    for label, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} samples ({percentage:.2f}%)")
    
    print(f"\n Visualization saved to: {output_path}")

def histogram(args):
    """
    Generate text length histogram
    
    Example:
        python main.py eda histogram --csv_path data.csv --text_col description --unit words
    """
    print(f"Analyzing text lengths in {args.csv_path}...")
    
    # Load data
    df = load_csv(args.csv_path)
    
    if args.text_col not in df.columns:
        print(f"Error: Column '{args.text_col}' not found in CSV", file=sys.stderr)
        return
    
    # Calculate lengths
    if args.unit == 'words':
        lengths = df[args.text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    else:
        lengths = df[args.text_col].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    # Generate visualization
    output_path = plot_histogram(lengths, args.unit, args.output)
    
    # Print statistics
    print(f"\n📈 Text Length Statistics ({args.unit}):")
    print(f"  Mean:   {lengths.mean():.2f}")
    print(f"  Median: {lengths.median():.2f}")
    print(f"  Std:    {lengths.std():.2f}")
    print(f"  Min:    {lengths.min()}")
    print(f"  Max:    {lengths.max()}")
    
    print(f"\n✅ Histogram saved to: {output_path}")

def wordcloud(args):
    """
    Generate word cloud visualization
    
    Example:
        python main.py eda wordcloud --csv_path data.csv --text_col description
    """
    print(f"☁️ Generating word cloud from {args.csv_path}...")
    
    # Load data
    df = load_csv(args.csv_path)
    
    if args.text_col not in df.columns:
        print(f"❌ Error: Column '{args.text_col}' not found in CSV", file=sys.stderr)
        return
    
    # Generate word cloud(s)
    output_path = generate_wordcloud(df, args.text_col, args.label_col, args.output)
    
    print(f"\n✅ Word cloud saved to: {output_path}")

def remove_outliers(args):
    """
    Remove statistical outliers based on text length (BONUS)
    
    Example:
        python main.py eda remove-outliers --csv_path data.csv --text_col description --output clean.csv
    """
    print(f"Detecting outliers in {args.csv_path}...")
    
    # Load data
    df = load_csv(args.csv_path)
    
    if args.text_col not in df.columns:
        print(f"Error: Column '{args.text_col}' not found in CSV", file=sys.stderr)
        return
    
    # Calculate text lengths
    lengths = df[args.text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    original_count = len(df)
    
    if args.method == 'iqr':
        Q1 = lengths.quantile(0.25)
        Q3 = lengths.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (lengths >= lower_bound) & (lengths <= upper_bound)
    else:  # zscore
        mean = lengths.mean()
        std = lengths.std()
        z_scores = (lengths - mean) / std
        mask = abs(z_scores) < 3
    
    df_clean = df[mask]
    removed_count = original_count - len(df_clean)
    
    # Save cleaned data
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.output, index=False)
    
    print(f"\nOutlier Removal Results:")
    print(f"  Original samples: {original_count}")
    print(f"  Removed samples:  {removed_count}")
    print(f"  Remaining samples: {len(df_clean)}")
    print(f"  Removal rate: {(removed_count/original_count)*100:.2f}%")
    
    print(f"\n✅ Clean data saved to: {args.output}")

def top_words(args):
    """
    Analyze most frequent words per class
    
    Example:
        python main.py eda top-words --csv_path data.csv --text_col text --label_col class
    """
    print(f"Analyzing top {args.top_n} words per class...")
        
    df = load_csv(args.csv_path)
    
    if args.text_col not in df.columns or args.label_col not in df.columns:
        print(f"Error: Required columns not found", err=True)
        return
    
    output_path = plot_top_words_per_class(df, args.text_col, args.label_col, args.top_n, args.output)
    print(f"\nTop words visualization saved to: {output_path}")

def ngrams_per_class(args):
    df = load_csv(args.csv_path)
    
    if args.text_col not in df.columns or args.label_col not in df.columns:
        print(f"Error: Required columns not found", err=True)
        return
    
    output_path = plot_ngrams_per_class(df, args.text_col, args.label_col, args.output)
    print(f"\ngrams per class saved to: {output_path}")

def statista(args):
    df = load_csv(args.csv_path)
    
    if args.text_col not in df.columns:
        print(f"Error: Required columns not found", err=True)
        return
    
    dict = get_text_stats(df, args.text_col)
    with open(args.output, 'w') as file:
        for key, value in dict.items():
            file.write(f"{key}: {value}\n")
    print(f"\nstatista saved into: {args.output}")
    

def eda_main(args):
    """Main entry point for EDA commands"""
    if args.eda_command == 'distribution':
        distribution(args)
    elif args.eda_command == 'histogram':
        histogram(args)
    elif args.eda_command == 'wordcloud':
        wordcloud(args)
    elif args.eda_command == 'remove-outliers':
        remove_outliers(args)
    elif args.eda_command == 'top-words':
        top_words(args)
    elif args.eda_command == 'ngrams':
        ngrams_per_class(args)
    elif args.eda_command == 'statista':
        statista(args)
    else:
        print("Error: No EDA subcommand specified")
        print("Available subcommands: distribution, histogram, wordcloud, top-words, ngrams, remove-outliers")
        sys.exit(1)
