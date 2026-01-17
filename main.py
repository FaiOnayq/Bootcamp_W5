import argparse
import sys
from commands.eda import eda_main
from commands.preprocessing import preprocess_main
from commands.embedding import embed_main
from commands.training import train_main
from commands.pipeline import pipeline_main
from commands.generate import generate_main

def main():
    parser = argparse.ArgumentParser(
        description='Arabic NLP Classification CLI Tool\nA complete pipeline for Arabic text classification: EDA → Preprocessing → Embedding → Training',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-v','--version', action='version', version='1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # EDA subparser
    eda_parser = subparsers.add_parser('eda', help='Exploratory Data Analysis commands')
    eda_subparsers = eda_parser.add_subparsers(dest='eda_command', help='EDA subcommands')
    
    # EDA distribution command
    dist_parser = eda_subparsers.add_parser('distribution', help='Visualize class distribution in the dataset')
    dist_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    dist_parser.add_argument('--label_col', required=True, help='Label column name')
    dist_parser.add_argument('--plot_type', default='pie', choices=['pie', 'bar'], help='Chart type')
    dist_parser.add_argument('--output', default='outputs/visualizations/distribution.png', help='Output path')
    
    # EDA histogram command
    hist_parser = eda_subparsers.add_parser('histogram', help='Generate text length histogram')
    hist_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    hist_parser.add_argument('--text_col', required=True, help='Text column name')
    hist_parser.add_argument('--unit', default='words', choices=['words', 'chars'], help='Count unit')
    hist_parser.add_argument('--output', default='outputs/visualizations/histogram.png', help='Output path')
    
    # EDA wordcloud command
    wc_parser = eda_subparsers.add_parser('wordcloud', help='Generate word cloud visualization')
    wc_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    wc_parser.add_argument('--text_col', required=True, help='Text column name')
    wc_parser.add_argument('--label_col', required=True, help='Label column (for per-class wordclouds)')
    wc_parser.add_argument('--output', default='outputs/visualizations', help='Output path')
    
    # EDA top words in class
    topwords_parser = eda_subparsers.add_parser('top-words', help='Analyze most frequent words per class')
    topwords_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    topwords_parser.add_argument('--text_col', required=True, help='Text column name')
    topwords_parser.add_argument('--label_col', required=True, help='Label column name')
    topwords_parser.add_argument('--top_n', type=int, default=20, help='Number of top words to display per class (default: 20)')
    topwords_parser.add_argument('--output', default='outputs/visualizations', help='Output directory path')

    # For ngrams_per_class function
    ngrams_parser = eda_subparsers.add_parser('ngrams', help='Analyze n-grams per class')
    ngrams_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    ngrams_parser.add_argument('--text_col', required=True, help='Text column name')
    ngrams_parser.add_argument('--label_col', required=True, help='Label column name')
    ngrams_parser.add_argument('--n', type=int, default=2, choices=[2, 3], help='N-gram size (2 for bigrams, 3 for trigrams, default: 2)')
    ngrams_parser.add_argument('--top_n', type=int, default=15, help='Number of top n-grams to display per class (default: 15)')
    ngrams_parser.add_argument('--output', default='outputs/visualizations', help='Output directory path')
    
    # text statisc
    ngrams_parser = eda_subparsers.add_parser('statista', help='Analyze text statisc')
    ngrams_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    ngrams_parser.add_argument('--text_col', required=True, help='Text column name')
    ngrams_parser.add_argument('--output', default='outputs/visualizations/statista.txt', help='Output directory path')
    
    
    # Preprocessing command placeholder
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocessing commands')
    preprocess_subparsers = preprocess_parser.add_subparsers(dest='preprocess_command', help='Preprocessing subcommands')

    # remove
    remove_p = preprocess_subparsers.add_parser('remove', help='Remove Arabic noise')
    remove_p.add_argument('--csv_path', required=True)
    remove_p.add_argument('--text_col', required=True)
    remove_p.add_argument('--output', required=True)
    remove_p.add_argument('--remove', default='all')

    # stopwords
    stop_p = preprocess_subparsers.add_parser('stopwords', help='Remove stopwords')
    stop_p.add_argument('--csv_path', required=True)
    stop_p.add_argument('--text_col', required=True)
    stop_p.add_argument('--sw_path', default="./assets/stopwords.txt", help="Path to text file of stopwords")
    stop_p.add_argument('--output', required=True)

    # replace
    replace_p = preprocess_subparsers.add_parser('replace', help='Normalize Arabic text')
    replace_p.add_argument('--csv_path', required=True)
    replace_p.add_argument('--text_col', required=True)
    replace_p.add_argument('--output', required=True)

    # all
    all_p = preprocess_subparsers.add_parser('all', help='Run full preprocessing pipeline')
    all_p.add_argument('--csv_path', required=True)
    all_p.add_argument('--text_col', required=True)
    all_p.add_argument('--sw_path', default="./assets/stopwords.txt", help="Path to text file of stopwords")
    all_p.add_argument('--output', required=True)
    all_p.add_argument('--stemmer', default='snowball', choices=['snowball', 'isri'])
    all_p.add_argument('--remove', default='all')

    # stem
    stem_p = preprocess_subparsers.add_parser('stem', help='Apply stemming')
    stem_p.add_argument('--csv_path', required=True)
    stem_p.add_argument('--text_col', required=True)
    stem_p.add_argument('--output', required=True)
    stem_p.add_argument('--stemmer', default='snowball', choices=['snowball', 'isri'])

    # lemmatize
    lemma_p = preprocess_subparsers.add_parser('lemmatize', help='Apply lemmatization')
    lemma_p.add_argument('--csv_path', required=True)
    lemma_p.add_argument('--text_col', required=True)
    lemma_p.add_argument('--output', required=True)
    
        
    # Embedding command placeholder
    embed_parser = subparsers.add_parser('embed', help='Embedding commands')
    embed_subparsers = embed_parser.add_subparsers(dest='embed_command', help='Embedding subcommands')

    # tfidf
    tfidf_p = embed_subparsers.add_parser('tfidf', help='TF-IDF embeddings')
    tfidf_p.add_argument('--csv_path', required=True)
    tfidf_p.add_argument('--text_col', required=True)
    tfidf_p.add_argument('--max_features', type=int, default=5000)
    tfidf_p.add_argument('--ngram_range', default='1,1')
    tfidf_p.add_argument('--output', required=True)

    # model2vec
    m2v_p = embed_subparsers.add_parser('model2vec', help='Model2Vec embeddings')
    m2v_p.add_argument('--csv_path', required=True)
    m2v_p.add_argument('--text_col', required=True)
    m2v_p.add_argument('--output', required=True)

    # bert
    bert_p = embed_subparsers.add_parser('bert', help='BERT embeddings')
    bert_p.add_argument('--csv_path', required=True)
    bert_p.add_argument('--text_col', required=True)
    bert_p.add_argument('--output', required=True)

    # word2vec
    w2v_p = embed_subparsers.add_parser('word2vec', help='Word2Vec embeddings')
    w2v_p.add_argument('--csv_path', required=True)
    w2v_p.add_argument('--text_col', required=True)
    w2v_p.add_argument('--vector_size', type=int, default=100)
    w2v_p.add_argument('--window', type=int, default=5)
    w2v_p.add_argument('--output', required=True)

    # fasttext
    ft_p = embed_subparsers.add_parser('fasttext', help='FastText embeddings')
    ft_p.add_argument('--csv_path', required=True)
    ft_p.add_argument('--text_col', required=True)
    ft_p.add_argument('--vector_size', type=int, default=100)
    ft_p.add_argument('--window', type=int, default=5)
    ft_p.add_argument('--output', required=True)

    
    # Training command placeholder
    train_parser = subparsers.add_parser('train', help='Training commands')
    train_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    train_parser.add_argument('--embedding_path', required=True, help='Path to embeddings pickle')
    train_parser.add_argument('--label_col', required=True, help='Label column name')
    train_parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    train_parser.add_argument('--models', default='knn,lr,rf', help='Models to train (comma-separated)')
    train_parser.add_argument('--output_dir', default='outputs/', help='Report output directory')

    
    # Pipeline command placeholder
    pipeline_parser = subparsers.add_parser('pipeline', help='Pipeline commands')

    pipeline_parser.add_argument('--csv_path', required=True, help='Path to CSV file')
    pipeline_parser.add_argument('--text_col', required=True, help='Text column name')
    pipeline_parser.add_argument('--label_col', required=True, help='Label column name')
    pipeline_parser.add_argument(
        '--preprocessing',
        default='all',
        help='Preprocessing steps (comma-separated)[remove,stopwords,replace,stem,lemmatize,all]'
    )
    pipeline_parser.add_argument('--sw_path', default="./assets/stopwords.txt", help="Path to text file of stopwords")
    pipeline_parser.add_argument('--stemmer', default='snowball', choices=['snowball', 'isri'])

    pipeline_parser.add_argument(
        '--embedding',
        default='tfidf',
        choices=['tfidf', 'model2vec', 'bert', 'word2vec', 'fasttext'],
        help='Embedding method'
    )
    pipeline_parser.add_argument('--ngram_range', default='1,1')
    pipeline_parser.add_argument('--max_features', type=int, default=5000)


    pipeline_parser.add_argument(
        '--training',
        default='knn,lr,rf',
        help='Models to train (comma-separated)'
    )
    pipeline_parser.add_argument('--test_size', type=float, default=0.2)
    pipeline_parser.add_argument('--output', default='outputs/pipeline')

    # generate
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--model', default='gemini', choices=['gemini', 'openai', 'local'])
    gen_parser.add_argument('--api_key', required=True)
    gen_parser.add_argument('--class_name', required=True)
    gen_parser.add_argument('--count', type=int, default=100)
    gen_parser.add_argument('--prompt')
    gen_parser.add_argument('--output', required=True)
    gen_parser.add_argument('--temperature', type=float, default=0.9)
    gen_parser.add_argument('--append', default=True,  help='Append to existing file if it exists')

    
    args = parser.parse_args()
    
    if args.command == 'eda':
        eda_main(args)
    elif args.command == 'preprocess':
        preprocess_main(args)
    elif args.command == 'embed':
        embed_main(args)
    elif args.command == 'train':
        train_main(args)
    elif args.command == 'pipeline':
        pipeline_main(args)
    elif args.command == 'generate':
        generate_main(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()