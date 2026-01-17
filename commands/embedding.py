
import pickle
from pathlib import Path

from utils.data_handler import load_csv
from utils.embeddings import (
    create_tfidf_embeddings,
    create_model2vec_embeddings,
    create_bert_embeddings,
    create_word2vec_embeddings,
    create_fasttext_embeddings
)


def embed_main(args):
    if args.embed_command == 'tfidf':
        tfidf_cmd(args)
    elif args.embed_command == 'model2vec':
        model2vec_cmd(args)
    elif args.embed_command == 'bert':
        bert_cmd(args)
    elif args.embed_command == 'word2vec':
        word2vec_cmd(args)
    elif args.embed_command == 'fasttext':
        fasttext_cmd(args)
    else:
        raise ValueError("Unknown embed command")


def _load_and_check(csv_path, text_col):
    df = load_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV")
    return df


def _save_pickle(obj, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump(obj, f)


def _print_dense_stats(vectors):
    memory_mb = vectors.nbytes / (1024 * 1024)
    print("\n Embedding Statistics:")
    print(f"  Shape: {vectors.shape}")
    print(f"  Dimension: {vectors.shape[1]}")
    print(f"  Samples: {vectors.shape[0]}")
    print(f"  Memory: {memory_mb:.2f} MB")



def tfidf_cmd(args):
    print(f"Creating TF-IDF embeddings from {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)
    n_min, n_max = map(int, args.ngram_range.split(','))

    vectors, vectorizer = create_tfidf_embeddings(
        df[args.text_col].tolist(),
        max_features=args.max_features,
        ngram_range=(n_min, n_max)
    )

    _save_pickle(
        {'vectors': vectors, 'vectorizer': vectorizer},
        args.output
    )

    memory_mb = vectors.data.nbytes / (1024 * 1024)

    print("\n Embedding Statistics:")
    print(f"  Shape: {vectors.shape}")
    print(f"  Features: {vectors.shape[1]}")
    print(f"  Samples: {vectors.shape[0]}")
    print(f"  Memory: {memory_mb:.2f} MB")
    print(
        f"  Sparsity: "
        f"{(1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1])) * 100:.2f}%"
    )

    print(f"\nEmbeddings saved to: {args.output}")


def model2vec_cmd(args):
    print(f"Creating Model2Vec embeddings from {args.csv_path}...")
    model = "JadwalAlmaa/model2vec-ARBERTv2"
    print(f"   Using model: {model}")

    df = _load_and_check(args.csv_path, args.text_col)
    vectors = create_model2vec_embeddings(df[args.text_col].tolist(), model)

    _save_pickle(
        {'vectors': vectors, 'model_name': model},
        args.output
    )

    _print_dense_stats(vectors)
    print(f"\nEmbeddings saved to: {args.output}")


def bert_cmd(args):
    print(f"Creating BERT embeddings from {args.csv_path}...")
    model = "aubmindlab/bert-base-arabertv2"
    print(f"   Using model: {model}")

    df = _load_and_check(args.csv_path, args.text_col)
    vectors = create_bert_embeddings(df[args.text_col].tolist(), model)

    _save_pickle(
        {'vectors': vectors, 'model_name': model},
        args.output
    )

    _print_dense_stats(vectors)
    print(f"\n Embeddings saved to: {args.output}")


def word2vec_cmd(args):
    print(f"Creating Word2Vec embeddings from {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)
    vectors, model = create_word2vec_embeddings(
        df[args.text_col].tolist(),
        vector_size=args.vector_size,
        window=args.window
    )

    _save_pickle(
        {'vectors': vectors, 'model': model},
        args.output
    )

    _print_dense_stats(vectors)
    print(f"\nEmbeddings saved to: {args.output}")


def fasttext_cmd(args):
    print(f"Creating FastText embeddings from {args.csv_path}...")

    df = _load_and_check(args.csv_path, args.text_col)
    vectors, model = create_fasttext_embeddings(
        df[args.text_col].tolist(),
        vector_size=args.vector_size,
        window=args.window
    )

    _save_pickle(
        {'vectors': vectors, 'model': model},
        args.output
    )

    _print_dense_stats(vectors)
    print(f"\nEmbeddings saved to: {args.output}")
