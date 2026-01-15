"""
Training Commands
Train and evaluate classification models
"""
import warnings
warnings.filterwarnings('ignore', message='Resorting to unclean kill browser')

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

from utils.data_handler import load_csv
from utils.models import train_models, parse_model_specs
from utils.metrics import generate_report


def train_main(args):
    print(f"Training models on {args.csv_path}...")
    
    report_metadata = {
        'dataset': Path(args.csv_path).name,
        'embedding': Path(args.embedding_path).name,
        'label_column': args.label_col,
        'test_size': args.test_size,
        'models': args.models,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Load data
    df = load_csv(args.csv_path)
    
    # Add basic dataset stats to report
    dataset_stats = {
        'total_samples': len(df),
        'classes': df[args.label_col].nunique(),
        'class_distribution': df[args.label_col].value_counts().to_dict()
    }
    report_metadata.update(dataset_stats)

    if args.label_col not in df.columns:
        raise ValueError(f"Column '{args.label_col}' not found in CSV")

    # Load embeddings
    print(f"Loading embeddings from {args.embedding_path}...")
    with open(args.embedding_path, 'rb') as f:
        embedding_data = pickle.load(f)

    if 'vectors' not in embedding_data:
        raise ValueError("No 'vectors' key found in embedding file")

    X = embedding_data['vectors']
    if hasattr(X, 'toarray'):  # sparse → dense
        X = X.toarray()

    y = df[args.label_col].values

    # Parse models
    model_specs = parse_model_specs(args.models)

    print("\n Configuration:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {(np.unique(y))}")
    print(f"  Test size: {args.test_size * 100:.1f}%")
    print(f"  Models: {', '.join([m['name'] for m in model_specs])}")

    # Train
    print("\n Training models...")
    results = train_models(X, y, model_specs, args.test_size)

    # Results
    print("\n Results:")
    for model_name, metrics in results.items():
        print(f"\n  {model_name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")

    # Best model
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_model_name]['model']

    print(
        f"\n Best Model: {best_model_name} "
        f"(F1: {results[best_model_name]['f1']:.4f})"
    )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"model_{timestamp}"
    model_path = save_path / f"{best_model_name.lower().replace(' ', '_')}_best_model.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    
    with open(model_path, 'wb') as f:
        pickle.dump(
            {
                'model': best_model,
                'model_name': best_model_name,
                'metrics': results[best_model_name]
            },
            f
        )
    print(f"\nBest model saved to: {model_path}")

    # Generate report
    report_path = Path(save_path) / f"training_report_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n Generating report...")
    generate_report(
        results=results,
        dataset_info={
            'total_samples': len(X),
            'features': X.shape[1],
            'classes': len(np.unique(y)),
            'test_size': args.test_size,
            'metadata': report_metadata,  # Add this line
            'embedding_file': Path(args.embedding_path).name,
            'dataset_file': Path(args.csv_path).name
        },
        output_path=str(report_path),
        best_model_name=best_model_name
    )

    print(f"\n Report saved to: {report_path}")
