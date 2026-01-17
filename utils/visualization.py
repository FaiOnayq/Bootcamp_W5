import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import numpy as np
from wordcloud import WordCloud
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
from io import BytesIO
import os
# Set Plotly template
import plotly.io as pio
pio.templates.default = "plotly_white"
import plotly.graph_objects as go
import plotly.offline as pyo
from pathlib import Path

def plot_distribution(df, label_col, plot_type='pie', output_path='outputs/visualizations/distribution.png'):
    """
    Plot class distribution using Plotly
    
    Args:
        df: DataFrame
        label_col: Column name for labels
        plot_type: 'pie' or 'bar'
        output_path: Output file path
    
    Returns:
        Path to saved plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ['Class', 'Count']
    
    if plot_type == 'pie':
        fig = go.Figure(data=[go.Pie(
            labels=counts['Class'],
            values=counts['Count'],
            hole=0.3,
            textinfo='label+percent',
            textposition='inside',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title=dict(
                text=f'Class Distribution - {label_col}',
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=True,
            height=600,
            width=800
        )
        
    else:  # bar chart
        fig = px.bar(
            counts,
            x='Class',
            y='Count',
            color='Class',
            color_discrete_sequence=px.colors.qualitative.Set3,
            text='Count'
        )
        
        fig.update_layout(
            title=dict(
                text=f'Class Distribution - {label_col}',
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="Class",
            yaxis_title="Count",
            showlegend=False,
            height=600,
            width=800
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_xaxes(tickangle=45)
    
    fig.write_image(output_path, width=1200, height=800)
    
    return output_path

# def plot_histogram(lengths, unit='words', output_path='outputs/visualizations/histogram.html'):
    """
    Plot text length histogram using Plotly
    
    Args:
        lengths: Series or list of text lengths
        unit: 'words' or 'chars'
        output_path: Output file path
    
    Returns:
        Path to saved plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to Series if not already
    if not isinstance(lengths, pd.Series):
        lengths = pd.Series(lengths)
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=lengths,
        nbinsx=50,
        name='Distribution',
        marker_color='skyblue',
        opacity=0.7,
        hovertemplate='Length: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_val = lengths.mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )
    
    # Add median line
    median_val = lengths.median()
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top right"
    )
    
    # Add box plot as inset
    fig.add_trace(go.Box(
        x=lengths,
        name='Box Plot',
        marker_color='lightgray',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Text Length Distribution ({unit})',
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title=f'Length ({unit})',
        yaxis_title='Frequency',
        barmode='overlay',
        height=600,
        width=900,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Update box plot to be less prominent
    fig.data[1].update(xaxis='x2')
    fig.update_layout(xaxis2=dict(
        domain=[0.7, 0.95],
        anchor='y2'
    ))
    fig.update_layout(yaxis2=dict(
        domain=[0.7, 0.95],
        anchor='x2'
    ))
    
    # Save as HTML for interactivity
    fig.write_html(output_path)
    
    # Also save as static image
    static_path = output_path.replace('.html', '.png')
    fig.write_image(static_path, width=1200, height=800)
    
    return output_path

def plot_histogram(lengths, unit='words', output_path='outputs/visualizations/histogram.png'):
    """
    Plot text length histogram using Plotly

    Args:
        lengths: Series or list of text lengths
        unit: 'words' or 'chars'
        output_path: Output file path (Plotly saves as HTML by default)

    Returns:
        Path to saved plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=lengths,
        nbinsx=50,
        marker=dict(
            color='rgba(0, 123, 255, 0.7)',
            line=dict(color='black', width=1)
        ),
        name='Distribution'
    ))

    mean_val = float(sum(lengths) / len(lengths)) if hasattr(lengths, '__len__') else float(lengths.mean())
    median_val = float(sorted(lengths)[len(lengths)//2]) if not hasattr(lengths, 'median') else float(lengths.median())

    fig.add_vline(
        x=mean_val,
        line=dict(color='red', dash='dash', width=2),
        annotation_text=f'Mean: {mean_val:.2f}',
        annotation_position="top"
    )

    fig.add_vline(
        x=median_val,
        line=dict(color='green', dash='dash', width=2),
        annotation_text=f'Median: {median_val:.2f}',
        annotation_position="top"
    )

    fig.update_layout(
        title=f'Text Length Distribution ({unit})',
        xaxis_title=f'Length ({unit})',
        yaxis_title='Frequency',
        bargap=0.1,
        hovermode='x unified',
        template='plotly_white'
    )

    fig.write_image(output_path, width=1200, height=800)
    

    return output_path

def generate_wordcloud(df, text_col, label_col, output_path='outputs/visualizations/'):
    """
    Generate word cloud visualization
    
    Args:
        df: DataFrame
        text_col: Text column name
        label_col: Optional label column for per-class wordclouds
        output_path: Output file path
    
    Returns:
        Path to saved plot
    """
    output_path = Path(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
    labels = df[label_col].unique()
    
    for label in labels:
        text = " ".join(df[df[label_col] == label][text_col].astype(str))
        
        reshaped_text = reshape(text)
        bidi_text = get_display(reshaped_text)
        
        FONT_PATH = (Path(__file__).resolve().parent.parent / "assets" / "NotoSansArabic-Regular.ttf")
        
        wordcloud = WordCloud(
            font_path=FONT_PATH, 
            width=800, 
            height=400, 
            background_color='white'
        ).generate(bidi_text)
        print(output_path)
        wordcloud.to_file(output_path / f"wordcloud_{label}.png")
    
    return output_path

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_confusion_matrix(cm, class_names, output_path='outputs/visualizations/confusion_matrix.png'):
    """
    Plot confusion matrix using Matplotlib (no browser needed).
    
    Args:
        cm: Confusion matrix array (2D numpy array)
        class_names: List of class names (strings)
        output_path: Output file path (e.g., .png)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted Label',
        ylabel='True Label',
        title='Confusion Matrix'
    )

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Prevent memory leaks in loops

    return output_path

def plot_feature_importance(importance, feature_names, top_n=20, output_path='outputs/visualizations/feature_importance.html'):
    """
    Plot feature importance using Plotly
    
    Args:
        importance: Array of importance values
        feature_names: List of feature names
        top_n: Number of top features to show
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get top N features
    indices = np.argsort(importance)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': top_features,
        'Importance': top_importance
    }).sort_values('Importance')
    
    # Create horizontal bar chart
    fig = px.bar(
        importance_df,
        y='Feature',
        x='Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        text='Importance'
    )
    
    fig.update_traces(
        texttemplate='%{text:.4f}',
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5
    )
    
    fig.update_layout(
        title=dict(
            text=f'Top {top_n} Most Important Features',
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        width=900,
        showlegend=False,
        coloraxis_showscale=False
    )
    
    # Add interactive features
    fig.update_layout(
        hovermode='y',
        yaxis=dict(
            tickmode='linear',
            categoryorder='total ascending'
        )
    )
    
    # Save as HTML for interactivity
    fig.write_html(output_path)
    
    # Also save as static image
    static_path = output_path.replace('.html', '.png')
    fig.write_image(static_path, width=1200, height=800)
    
    return output_path

def plot_training_history(history, metrics=['loss', 'accuracy'], output_path='outputs/visualizations/training_history.html'):
    """
    Plot training history (for neural networks) using Plotly
    
    Args:
        history: Training history object or dict
        metrics: List of metrics to plot
        output_path: Output file path
    
    Returns:
        Path to saved plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    n_metrics = len(metrics)
    fig = make_subplots(
        rows=n_metrics, 
        cols=1,
        subplot_titles=[f'{metric.title()} over Epochs' for metric in metrics]
    )
    
    for idx, metric in enumerate(metrics):
        row = idx + 1
        
        if metric in history_dict:
            # Plot training metric
            fig.add_trace(
                go.Scatter(
                    y=history_dict[metric],
                    mode='lines+markers',
                    name=f'Training {metric}',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ),
                row=row, col=1
            )
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                fig.add_trace(
                    go.Scatter(
                        y=history_dict[val_metric],
                        mode='lines+markers',
                        name=f'Validation {metric}',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    ),
                    row=row, col=1
                )
        
        fig.update_xaxes(title_text='Epoch', row=row, col=1)
        fig.update_yaxes(title_text=metric.title(), row=row, col=1)
    
    fig.update_layout(
        title_text='Training History',
        height=300 * n_metrics,
        width=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save as HTML for interactivity
    fig.write_html(output_path)
    
    # Also save as static image
    static_path = output_path.replace('.html', '.png')
    fig.write_image(static_path, width=1200, height=400 * n_metrics)
    
    return output_path


def plot_top_words_per_class(df, text_col, label_col, top_n=20, output_path='outputs/visualizations/top_words.png'):
    """
    Plot most frequent words per class
    
    Args:
        df: DataFrame
        text_col: Text column name
        label_col: Label column name
        top_n: Number of top words
        output_path: Output file path
    
    Returns:
        Path to saved plot
    """
    from collections import Counter
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    classes = df[label_col].unique()
    n_classes = len(classes)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 6))
    if n_classes == 1:
        axes = [axes]
    
    for idx, cls in enumerate(classes):
        # Get all words for this class
        class_texts = df[df[label_col] == cls][text_col]
        all_words = ' '.join(class_texts.astype(str)).split()
        
        # Count frequencies
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)
        
        if top_words:
            words, counts = zip(*top_words)
            
            axes[idx].barh(range(len(words)), counts)
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].set_xlabel('Frequency')
            axes[idx].set_title(f'Top {top_n} Words - {cls}')
            axes[idx].invert_yaxis()
    
    plt.tight_layout()
    output_path = output_path + '/top_words.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_ngrams_per_class(df, text_col, label_col, n=2, top_k=15, output_path='outputs/visualizations/ngrams.png'):
    """
    Plot most common n-grams per class
    
    Args:
        df: DataFrame
        text_col: Text column name
        label_col: Label column name
        n: N-gram size
        top_k: Number of top n-grams
        output_path: Output file path
    
    Returns:
        Path to saved plot
    """
    from collections import Counter
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_ngrams(text, n):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    classes = df[label_col].unique()
    n_classes = len(classes)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 6))
    if n_classes == 1:
        axes = [axes]
    
    for idx, cls in enumerate(classes):
        # Get all n-grams for this class
        class_texts = df[df[label_col] == cls][text_col]
        all_ngrams = []
        for text in class_texts.astype(str):
            all_ngrams.extend(get_ngrams(text, n))
        
        # Count frequencies
        ngram_counts = Counter(all_ngrams)
        top_ngrams = ngram_counts.most_common(top_k)
        
        if top_ngrams:
            ngrams, counts = zip(*top_ngrams)
            
            axes[idx].barh(range(len(ngrams)), counts)
            axes[idx].set_yticks(range(len(ngrams)))
            axes[idx].set_yticklabels(ngrams, fontsize=8)
            axes[idx].set_xlabel('Frequency')
            axes[idx].set_title(f'Top {top_k} {n}-grams - {cls}')
            axes[idx].invert_yaxis()
    
    plt.tight_layout()
    output_path = output_path + "/ngrams.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path