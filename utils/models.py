import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def parse_model_specs(models_str):
    """
    Parse model specifications string
    
    Args:
        models_str: Comma-separated model names with optional params
                   e.g., "knn:n_neighbors=7,lr:C=0.5,rf"
    
    Returns:
        List of model specification dicts
    """
    specs = []
    
    model_mapping = {
        'knn': ('K-Nearest Neighbors', KNeighborsClassifier, {'n_neighbors': 5}),
        'lr': ('Logistic Regression', LogisticRegression, {'max_iter': 1000, 'random_state': 42}),
        'rf': ('Random Forest', RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}),
        'svm': ('Support Vector Machine', SVC, {'kernel': 'rbf', 'random_state': 42}),
        'nb': ('Naive Bayes', MultinomialNB, {}),
        'dt': ('Decision Tree', DecisionTreeClassifier, {'random_state': 42}),
        'gb': ('Gradient Boosting', GradientBoostingClassifier, {'random_state': 42})
    }
    
    if models_str.lower() == 'all':
        models_str = ','.join(model_mapping.keys())
    
    for model_spec in models_str.split(','):
        model_spec = model_spec.strip()
        
        if ':' in model_spec:
            # Parse custom parameters
            model_name, params_str = model_spec.split(':', 1)
            params = {}
            for param in params_str.split(':'):
                key, value = param.split('=')
                # Try to convert to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                params[key] = value
        else:
            model_name = model_spec
            params = {}
        
        if model_name in model_mapping:
            name, clf_class, default_params = model_mapping[model_name]
            # Merge default and custom params
            final_params = {**default_params, **params}
            specs.append({
                'name': name,
                'class': clf_class,
                'params': final_params
            })
    
    return specs

def train_models(X, y, model_specs, test_size=0.2, random_state=42):
    """
    Train multiple models and evaluate them
    
    Args:
        X: Feature matrix
        y: Target labels
        model_specs: List of model specification dicts
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Dictionary with model results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    results = {}
    
    for spec in model_specs:
        model_name = spec['name']
        clf_class = spec['class']
        params = spec['params']
        
        print(f"  Training {model_name}...")
        
        # Create and train model
        clf = clf_class(**params)
        
        # Handle sparse matrices for Naive Bayes
        if clf_class == MultinomialNB and hasattr(X_train, 'toarray'):
            # Ensure non-negative values for NB
            X_train_nb = X_train.toarray()
            X_test_nb = X_test.toarray()
            X_train_nb = np.abs(X_train_nb)
            X_test_nb = np.abs(X_test_nb)
            clf.fit(X_train_nb, y_train)
            y_pred = clf.predict(X_test_nb)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle multiclass averaging
        average = 'weighted' if len(np.unique(y)) > 2 else 'binary'
        
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[model_name] = {
            'model': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    return results

def get_feature_importance(model, feature_names=None, top_n=20):
    """
    Extract feature importance from a trained model
    
    Args:
        model: Trained sklearn model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        Dictionary with feature importance info
    """
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        if len(model.coef_.shape) > 1:
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            importance = np.abs(model.coef_)
    
    if importance is not None and feature_names is not None:
        indices = np.argsort(importance)[-top_n:]
        top_features = [(feature_names[i], importance[i]) for i in indices]
        return {
            'importance': importance,
            'top_features': top_features
        }
    
    return None