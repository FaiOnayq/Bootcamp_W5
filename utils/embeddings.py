"""
Embedding Utilities
Create text embeddings using various methods
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_embeddings(texts, max_features=5000, ngram_range=(1, 1)):
    """
    Create TF-IDF embeddings
    
    Args:
        texts: List of text strings
        max_features: Maximum number of features
        ngram_range: Tuple of (min_n, max_n) for n-grams
    
    Returns:
        Sparse matrix of TF-IDF vectors, fitted vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95
    )
    
    vectors = vectorizer.fit_transform(texts)
    
    return vectors, vectorizer

def create_model2vec_embeddings(texts, model_name='JadwalAlmaa/model2vec-ARBERTv2'):
    """
    Create Model2Vec embeddings
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model name
    
    Returns:
        Numpy array of embeddings
    """
    try:
        from model2vec import StaticModel
        
        # Load model
        model = StaticModel.from_pretrained(model_name)
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=True)
        
        return np.array(embeddings)
    
    except ImportError:
        print("Error: model2vec not installed. Install with: pip install model2vec")
        raise
    except Exception as e:
        print(f"Error loading Model2Vec: {e}")
        print("Falling back to TF-IDF...")
        vectors, _ = create_tfidf_embeddings(texts)
        return vectors.toarray()

def create_bert_embeddings(texts, model_name='aubmindlab/bert-base-arabertv2'):
    """
    Create BERT embeddings (BONUS)
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model name
    
    Returns:
        Numpy array of embeddings
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        
        # Process in batches
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    except ImportError:
        print("Error: transformers not installed. Install with: pip install transformers torch")
        raise


def create_word2vec_embeddings(texts, vector_size=100, window=5, min_count=2):
    """
    Create Word2Vec embeddings (BONUS)
    
    Args:
        texts: List of text strings
        vector_size: Dimension of embeddings
        window: Context window size
        min_count: Minimum word frequency
    
    Returns:
        Numpy array of document embeddings, trained model
    """
    try:
        from gensim.models import Word2Vec
        
        # Tokenize texts
        tokenized = [text.split() for text in texts]
        
        # Train Word2Vec
        model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=10
        )
        
        # Create document embeddings by averaging word vectors
        embeddings = []
        for tokens in tokenized:
            valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
            if valid_vectors:
                doc_vector = np.mean(valid_vectors, axis=0)
            else:
                doc_vector = np.zeros(vector_size)
            embeddings.append(doc_vector)
        
        return np.array(embeddings), model
    
    except ImportError:
        print("Error: gensim not installed. Install with: pip install gensim")
        raise

def create_fasttext_embeddings(texts, vector_size=100, window=5, min_count=2):
    """
    Create FastText embeddings (BONUS)
    
    Args:
        texts: List of text strings
        vector_size: Dimension of embeddings
        window: Context window size
        min_count: Minimum word frequency
    
    Returns:
        Numpy array of document embeddings, trained model
    """
    try:
        from gensim.models import FastText
        
        # Tokenize texts
        tokenized = [text.split() for text in texts]
        
        # Train FastText
        model = FastText(
            sentences=tokenized,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=10
        )
        
        # Create document embeddings by averaging word vectors
        embeddings = []
        for tokens in tokenized:
            # FastText can handle OOV words
            valid_vectors = [model.wv[word] for word in tokens]
            if valid_vectors:
                doc_vector = np.mean(valid_vectors, axis=0)
            else:
                doc_vector = np.zeros(vector_size)
            embeddings.append(doc_vector)
        
        return np.array(embeddings), model
    
    except ImportError:
        print("Error: gensim not installed. Install with: pip install gensim")
        raise