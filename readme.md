
# 🐪 Arabic NLP CLI Toolkit  
> Transform Arabic text into insights with a single command

A streamlined CLI pipeline for Arabic text classification—from raw data to trained models in minutes.

## What It Does
This tool handles the complete Arabic NLP workflow:

`Raw Arabic Text → Clean Data → Smart Embeddings → Trained Models`

No notebooks. No complexity. Just results.

## Quick Start
Clone the repo:
```bash
git clone https://github.com/FaiOnayq/Bootcamp_W5.git
cd Bootcamp_W5
```
Install dependencies:
```bash
uv venv -p 3.11
uv pip install -r requirements.txt
```
Run the CLI via:

```bash
uv run main.py --help
```
Run code:
```bash
# Full pipeline in one command
uv run main.py pipeline `
  --csv_path ASA.csv `
  --text_col text `
  --label_col Sentiment `
  --preprocessing all `
  --embedding tfidf `
  --training knn,lr,rf

# Or go step-by-step
uv run main.py eda distribution --csv_path ASA.csv --label_col Sentiment
uv run main.py preprocess all --csv_path ASA.csv --text_col text --output clean.csv
uv run main.py embed tfidf --csv_path clean.csv --text_col text --output vectors.pkl
uv run main.py train --csv_path clean.csv --embedding_path vectors.pkl --label_col Sentiment
```


## Features 

- **Exploratory Data Analysis**
  - Class distribution
  - Text length histograms
  - Word clouds (per class)
  - Top words
  - Text statistics

- **Arabic Preprocessing**
  - Noise & diacritics removal
  - Stopword filtering
  - Normalization
  - Stemming (Snowball / ISRI)
  - Lemmatization
  - One-shot full pipeline

- **Multiple Embedding Strategies**
  - TF-IDF
  - Word2Vec
  - FastText
  - BERT
  - Model2Vec

- **Model Training & Evaluation**
  - Train multiple classifiers in one run
  - - knn – K-Nearest Neighbors
  - - lr – Logistic Regression
  - - rf – Random Forest
  - - svm – Support Vector Machine
  - - nb – Naive Bayes
  - - dt – Decision Tree
  - - gb – Gradient Boosting
  - Automatic train/test split
  - Performance reports

- **End-to-End Pipeline**
  - From CSV → trained models in one command

- **Synthetic Data Generation**
  - Generate labeled Arabic text using Gemini LLMs (one class per command)
```bash
uv run main.py generate `
  --model gemini `
  --api_key # need to add your key `
  --class_name "sports" `
  --count 10 `
  --output generated.csv
```
To add another class; use same output file!


## EDA Commands

### Class Distribution

```bash
uv run main.py eda distribution `
  --csv_path ASA.csv `
  --label_col Sentiment `
  --plot_type pie
```

### Text Length Histogram

```bash
uv run main.py eda histogram `
  --csv_path ASA.csv `
  --text_col text `
  --unit words
```

### Word Clouds (Per Class)
Use this visualization after Clean !!!
```bash
uv run main.py eda wordcloud `
  --csv_path ASA.csv `
  --text_col text `
  --label_col Sentiment
```

### Top Words Per Class

```bash
uv run main.py eda top-words `
  --csv_path ASA.csv `
  --text_col text `
  --label_col Sentiment `
  --top_n 20
```

### Text Statistics

```bash
uv run main.py eda statista `
  --csv_path ASA.csv `
  --text_col text
```

---

## Preprocessing Commands

### Remove Noise

```bash
uv run main.py preprocess remove `
  --csv_path ASA.csv `
  --text_col text `
  --output clean.csv
```


### Stemming

```bash
uv run main.py preprocess stem `
  --csv_path clean.csv `
  --text_col text `
  --stemmer snowball `
  --output clean.csv
```

### Lemmatize
```bash
uv run main.py preprocess lemmatize `
  --csv_path clean.csv `
  --text_col text `
  --output clean.csv
```

### Full Preprocessing Pipeline

```bash
uv run main.py preprocess all `
  --csv_path ASA.csv `
  --text_col text `
  --output clean.csv
```

---

## Embeddings

### TF-IDF

```bash
uv run main.py embed tfidf `
  --csv_path clean.csv `
  --text_col text `
  --output tfidf.pkl
```

### Word2Vec / FastText

```bash
uv run main.py embed word2vec --csv_path clean.csv --text_col text --output w2v.pkl
uv run main.py embed fasttext --csv_path clean.csv --text_col text --output ft.pkl
```

### BERT / Model2Vec

```bash
uv run main.py embed bert --csv_path clean.csv --text_col text --output bert.pkl
```

---

## Training Models

```bash
uv run main.py train `
  --csv_path clean.csv `
  --embedding_path tfidf.pkl `
  --label_col Sentiment `
  --models knn,lr,rf,svm `
  --output_dir outputs/
```

### Available Models

| Code  | Model                  |
| ----- | ---------------------- |
| `knn` | K-Nearest Neighbors    |
| `lr`  | Logistic Regression    |
| `rf`  | Random Forest          |
| `svm` | Support Vector Machine |
| `nb`  | Naive Bayes            |
| `dt`  | Decision Tree          |
| `gb`  | Gradient Boosting      |

---

## Full Pipeline (One Command)

```bash
uv run main.py pipeline `
  --csv_path ASA.csv `
  --text_col text `
  --label_col Sentiment `
  --embedding tfidf `
  --training knn,lr,rf
```

This runs:
1. EDA
2. Preprocessing
3. Embedding
4. Training
5. Evaluation

---

## Outputs

All results are saved under:

```text
pipelin/run_{timestamp}/
├── visualizations/
├── training_report_{timestamp}.md
├── model_best.pkl
```


<br>


---
Bootcamp: AI Professionals | Week 5 <br>
Date: 17/01/2025
