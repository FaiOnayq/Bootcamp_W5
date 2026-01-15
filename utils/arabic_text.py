import re
import string
import numpy as np

# Arabic character sets
ARABIC_TASHKEEL = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
ARABIC_TATWEEL = 'ـ'
ARABIC_PUNCTUATION = '،؛؟'

import stanza

_AR_PIPELINE = None

def get_ar_pipeline():
    global _AR_PIPELINE
    if _AR_PIPELINE is None:
        _AR_PIPELINE = stanza.Pipeline(
            lang="ar",
            processors="tokenize,mwt,pos,lemma",
            tokenize_no_ssplit=True,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES
        )
    return _AR_PIPELINE

ENGLISH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
}

def remove_arabic_noise(text, remove_types='all'):
    """
    Remove Arabic-specific noise from text
    
    Args:
        text: Input text string
        remove_types: Comma-separated types or 'all'
        Options: tashkeel, tatweel, tarqeem, links, special
    
    Returns:
        Cleaned text string
    """
    if not text:
        return ''
    
    types = remove_types.split(',') if remove_types != 'all' else ['tashkeel', 'tatweel', 'tarqeem', 'links', 'special', 'HTML']
    
    # Remove tashkeel (diacritics)
    if 'tashkeel' in types or 'all' in types:
        text = ARABIC_TASHKEEL.sub('', text)
    
    # Remove tatweel (kashida)
    if 'tatweel' in types or 'all' in types:
        text = text.replace(ARABIC_TATWEEL, '')
    
    # Remove numbers (tarqeem)
    if 'tarqeem' in types or 'all' in types:
        text = re.sub(r'\d+', '', text)
    
    # Remove URLs
    if 'links' in types or 'all' in types:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+\.com\S*|\S+\.org\S*|\S+\.net\S*', '', text)
    
    # Remove special characters (keep Arabic and spaces)
    if 'special' in types or 'all' in types:
        # Keep Arabic letters, English letters, and spaces
        text = re.sub(r'[^\u0600-\u06FFa-zA-Z\s]', ' ', text)
    
    if 'HTML' in types or 'all' in types:
        # Keep Arabic letters, English letters, and spaces
        text = re.sub('<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text, sw_path, language='ar'):
    """
    Remove stopwords from text
    
    Args:
        text: Input text string
        language: 'ar', 'en', or 'auto'
    
    Returns:
        Text without stopwords
    """
    if not text:
        return ''
    
    with open(sw_path, "r", encoding="utf-8") as f:
        ARABIC_STOPWORDS = {line.strip() for line in f}
    words = text.split()
    
    if language == 'ar':
        stopwords = ARABIC_STOPWORDS
    elif language == 'en':
        stopwords = ENGLISH_STOPWORDS
    elif language == 'auto':
        stopwords = ARABIC_STOPWORDS | ENGLISH_STOPWORDS
    else:
        stopwords = set()
    
    filtered_words = [word for word in words if word not in stopwords]
    
    return ' '.join(filtered_words)

def normalize_arabic(text):
    """
    Normalize Arabic text variants
    
    Normalizations:
    - Alef variants (أ، إ، آ) → ا
    - Alef Maqsura (ى) → ي
    - Taa Marbouta (ة) → ه
    - Hamza variants
    
    Args:
        text: Input text string
    
    Returns:
        Normalized text string
    """
    if not text:
        return ''
    
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub("چ", "ج", text)
    text = re.sub("پ", "ب", text)
    text = re.sub("ڜ", "ش", text)
    text = re.sub("ڪ", "ك", text)
    text = re.sub("ڧ", "ق", text)
    text = re.sub("ٱ", "ا", text)
    
    return text

def get_text_stats(df, column_name):
    """
    Calculate statistics for a list of texts
    
    Args:
        texts: List of text strings
    
    Returns:
        Dictionary with statistics
    """
    texts = df[column_name].fillna('').astype(str).tolist()
    
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]
    
    
    return {
        'total_texts': len(texts),
        'avg_words': np.mean(word_counts),
        'avg_chars': np.mean(char_counts),
        'median_words': np.median(word_counts),
        'median_chars': np.median(char_counts),
        'std_words': np.std(word_counts),
        'std_chars': np.std(char_counts),
        'min_words': np.min(word_counts),
        'max_words': np.max(word_counts),
        'min_chars': np.min(char_counts),
        'max_chars': np.max(char_counts)
    }

def stem_text(text, language='ar', stemmer='snowball'):
    """
    Args:
        text: Input text string
        language: Language code
        stemmer: 'snowball' or 'isri'
    
    Returns:
        Stemmed text
    """
    if not text:
        return ''
    
    try:
        if language == 'ar':
            if stemmer == 'isri':
                from nltk.stem.isri import ISRIStemmer
                stemmer_obj = ISRIStemmer()
            elif stemmer == "snowball":
                from nltk.stem.snowball import ArabicStemmer
                stemmer_obj = ArabicStemmer()
            else:
                print("Error: stemmer name is unknown")
                print("Available stemmer: isri, snowball")
                
        else:
            from nltk.stem.snowball import SnowballStemmer
            stemmer_obj = SnowballStemmer(language)
        
        words = text.split()
        stemmed = [stemmer_obj.stem(word) for word in words]
        return ' '.join(stemmed)
    
    except ImportError:
        print("Warning: NLTK not installed. Returning original text.")
        return text

def lemmatize_text(text, language='ar'):
    """    
    Args:
        text: Input text string
        language: Language code
    
    Returns:
        Lemmatized text
    """
    if not text:
        return ''
    
    try:
        if language == 'ar':
            nlp = get_ar_pipeline()
            doc = nlp(text)
            lemmas = []

            for sent in doc.sentences:
                for word in sent.words:
                    lemmas.append(word.lemma)

            text = " ".join(lemmas)
            return ARABIC_TASHKEEL.sub('', text)
        else:
            # Use spacy for other languages
            import spacy
            nlp = spacy.load(f'{language}_core_web_sm')
            doc = nlp(text)
            return ' '.join([token.lemma_ for token in doc])
    
    except ImportError:
        print("Warning: Required library not installed. Returning original text.")
        return text