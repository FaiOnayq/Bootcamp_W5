import numpy as np
from pathlib import Path
from datetime import datetime
from utils.visualization import plot_confusion_matrix

def generate_report(results, dataset_info, output_path, best_model_name):
    """
    Generate comprehensive training report in Markdown
    
    Args:
        results: Dictionary of model results
        dataset_info: Dataset information dict
        output_path: Output file path
        best_model_name: Name of best performing model
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append(f"# Training Report - {timestamp}\n")
    
    report.append("## Dataset Information\n")
    report.append(f"- **Dataset file**: {dataset_info.get('dataset_file', 'N/A')}\n")
    report.append(f"- **Embedding file**: {dataset_info.get('embedding_file', 'N/A')}\n")
    report.append(f"- **Total samples**: {dataset_info['total_samples']}\n")
    report.append(f"- **Features**: {dataset_info['features']}\n")
    report.append(f"- **Classes**: {dataset_info['classes']}\n")
    report.append(f"- **Test size**: {dataset_info['test_size'] * 100:.1f}%\n")
    
    # Add class distribution if available
    if 'metadata' in dataset_info and 'class_distribution' in dataset_info['metadata']:
        report.append("\n## Class Distribution\n")
        dist = dataset_info['metadata']['class_distribution']
        for cls, count in dist.items():
            report.append(f"- **{cls}**: {count} samples ({(count/dataset_info['total_samples'])*100:.1f}%)\n")
    
    report.append("\n")
    
    # Model Performance
    report.append("## Model Performance\n\n")
    
    # Create comparison table
    report.append("| Model | Accuracy | Precision | Recall | F1-Score |\n")
    report.append("|-------|----------|-----------|--------|----------|\n")
    
    for model_name, metrics in results.items():
        marker = " ⭐" if model_name == best_model_name else ""
        report.append(
            f"| {model_name}{marker} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} |\n"
        )
    
    report.append("\n")
    # Detailed results for each model
    report.append("## Confusion Results\n\n")
    
    viz_dir = Path(output_path).parent / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    for model_name, metrics in results.items():
        # Generate confusion matrix plot
        cm_filename = f"cm_{model_name.lower().replace(' ', '_')}.png"
        cm_path = viz_dir / cm_filename
        class_names = list(dataset_info['metadata']['class_distribution'].keys())
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, str(cm_path))
        
        report.append(f"### {model_name}\n\n")
        relative_path = f"./visualizations/{cm_filename}"
        report.append(f"![Confusion Matrix]({relative_path})\n\n")
        
        
    report.append("\n## Performance Summary\n\n")
    report.append(f"- **Total models evaluated**: {len(results)}\n")
    report.append(f"- **Best model**: {best_model_name}\n")
    
    best_metrics = results[best_model_name]
    report.append(f"This model achieved the highest F1-score of **{best_metrics['f1']:.4f}**.\n\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

def calculate_metrics_summary(results):
    """
    Calculate summary statistics across all models
    
    Args:
        results: Dictionary of model results
    
    Returns:
        Dictionary with summary statistics
    """
    metrics_list = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for model_name, metrics in results.items():
        metrics_list['accuracy'].append(metrics['accuracy'])
        metrics_list['precision'].append(metrics['precision'])
        metrics_list['recall'].append(metrics['recall'])
        metrics_list['f1'].append(metrics['f1'])
    
    summary = {}
    for metric_name, values in metrics_list.items():
        summary[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return summary