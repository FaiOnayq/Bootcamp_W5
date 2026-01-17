import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from utils.data_handler import load_csv
from utils.arabic_text import (
    remove_arabic_noise,
    remove_stopwords,
    normalize_arabic,
    stem_text
)
from utils.embeddings import (
    create_tfidf_embeddings,
    create_model2vec_embeddings,
    create_bert_embeddings,
    create_word2vec_embeddings,
    create_fasttext_embeddings
)
from utils.visualization import plot_distribution, plot_histogram, generate_wordcloud
from utils.arabic_text import get_text_stats



from utils.models import train_models, parse_model_specs
from utils.metrics import generate_report


def pipeline_main(args):
    print("Starting NLP Pipeline")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- LOAD DATA ---------------- #
    print(f"\nLoading data from {args.csv_path}...")
    
    report_metadata = {
        'dataset': Path(args.csv_path).name,
        'embedding': args.embedding,
        'label_column': args.label_col,
        'test_size': args.test_size,
        'models': args.training,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    df = load_csv(args.csv_path)
    
    dataset_stats = {
        'total_samples': len(df),
        'classes': df[args.label_col].nunique(),
        'class_distribution': df[args.label_col].value_counts().to_dict()
    }
    report_metadata.update(dataset_stats)

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found")

    if args.label_col not in df.columns:
        raise ValueError(f"Column '{args.label_col}' not found")

    print(f"   Total samples: {len(df)}")
    print(f"   Classes: {df[args.label_col].nunique()}")
    
    # -----------------EDA ----------------- #
    print(f"\nPerforming EDA...")
    plot_distribution(df, args.label_col,"pie", output_dir / "visualizations" / "class_distribution.png")
    print(f"   Class distribution plot saved to: {output_dir / 'visualizations' / 'class_distribution.png'}")
    lengths = df[args.text_col].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    plot_histogram(lengths, 'chars', output_dir / "visualizations" / "text_length_histogram.png")
    print(f"   Text length histogram saved to: {output_dir / 'visualizations' / 'text_length_histogram.png'}")  
    dict = get_text_stats(df, args.text_col)
    with open(output_dir/"stats.txt", 'w') as file:
        for key, value in dict.items():
            file.write(f"{key}: {value}\n")
    print(f"   statista saved into: {output_dir/'stats.txt'}")

    # ---------------- PREPROCESSING ---------------- #
    print(f"\nPreprocessing (steps: {args.preprocessing})...")
    steps = args.preprocessing.split(',')

    if 'remove' in steps or 'all' in steps:
        print("   1. Removing noise...")
        df[args.text_col] = df[args.text_col].apply(
            lambda x: remove_arabic_noise(str(x) if pd.notna(x) else '', 'all')
        )

    if 'stopwords' in steps or 'all' in steps:
        print("   2. Removing stopwords...")
        df[args.text_col] = df[args.text_col].apply(
            lambda x: remove_stopwords(str(x) if pd.notna(x) else '',args.sw_path)
        )

    if 'replace' in steps or 'all' in steps:
        print("   3. Normalizing text...")
        df[args.text_col] = df[args.text_col].apply(
            lambda x: normalize_arabic(str(x) if pd.notna(x) else '')
        )
    if 'stem' in steps or 'all' in steps:
        print("   4. Stemming text...")
        df[args.text_col] = df[args.text_col].apply(
            lambda x: stem_text(str(x) if pd.notna(x) else '', args.stemmer)
        )

    preprocessed_path = output_dir / "preprocessed_data.csv"
    df.to_csv(preprocessed_path, index=False)
    print(f"   Preprocessed data saved to: {preprocessed_path}")

    # ---------------- EMBEDDING ---------------- #
    print(f"\nCreating {args.embedding.upper()} embeddings...")

    texts = df[args.text_col].tolist()
    embedding_meta = {}

    if args.embedding == 'tfidf':
        n_min, n_max = map(int, args.ngram_range.split(','))

        X, vectorizer = create_tfidf_embeddings(
            texts,
            max_features=args.max_features,
            ngram_range=(n_min, n_max)
        )

        embedding_meta = {
            'method': 'tfidf',
            'vectorizer': vectorizer
        }

    elif args.embedding == 'model2vec':
        X = create_model2vec_embeddings(texts)

        embedding_meta = {
            'method': 'model2vec',
        }

    elif args.embedding == 'bert':
        X = create_bert_embeddings(texts)

        embedding_meta = {
            'method': 'bert',
        }

    elif args.embedding == 'word2vec':
        X, model = create_word2vec_embeddings(
            texts,
            vector_size=100,
            window=5
        )

        embedding_meta = {
            'method': 'word2vec',
            'model': model
        }

    elif args.embedding == 'fasttext':
        X, model = create_fasttext_embeddings(
            texts,
            vector_size=100,
            window=5
        )

        embedding_meta = {
            'method': 'fasttext',
            'model': model
        }

    else:
        raise ValueError(f"Unsupported embedding method: {args.embedding}")


    print(f"   Shape: {X.shape}")

    emb_path = output_dir / f"{args.embedding}_embeddings.pkl"
    with open(emb_path, 'wb') as f:
        pickle.dump({'vectors': X, **embedding_meta}, f)
    print(f"   Embeddings saved to: {emb_path}")

    # ---------------- TRAINING ---------------- #
    print(f"\nTraining models...")
    y = df[args.label_col].values
    model_specs = parse_model_specs(args.training)

    results = train_models(X, y, model_specs, args.test_size)

    print(f"\n Training Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")

    best_model_name = max(results, key=lambda k: results[k]['f1'])
    print(
        f"\n⭐ Best Model: {best_model_name} "
        f"(F1: {results[best_model_name]['f1']:.4f})"
    )

    # ---------------- SAVE MODELS ---------------- #
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for model_name, data in results.items():
        model_path = models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(
                {
                    'model': data['model'],
                    'metrics': {k: v for k, v in data.items() if k != 'model'}
                },
                f
            )

    print(f"\nModels saved to: {models_dir}")

    # ---------------- REPORT ---------------- #
    report_path = output_dir / f"pipeline_report.md"

    generate_report(
        results=results,
        dataset_info={
            'total_samples': X.shape[0],
            'features': X.shape[1],
            'classes': len(np.unique(y)),
            'test_size': args.test_size,
            'metadata': report_metadata,  # Add this line
            'dataset_file': Path(args.csv_path).name
        },
        output_path=str(report_path),
        best_model_name=best_model_name
    )

    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
