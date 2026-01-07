import pandas as pd
import numpy as np
import re
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD DATASET (JSONL FORMAT)
# ============================================
print("Loading dataset from JSONL...")

# Read JSONL file
data_list = []
with open('programming_problems.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))

df = pd.DataFrame(data_list)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("\n" + "="*50)
print("PREPROCESSING DATA")
print("="*50)

# Handle missing values
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')
df['input_description'] = df['input_description'].fillna('')
df['output_description'] = df['output_description'].fillna('')

# Keep original class labels (lowercase)
print(f"\n📊 Classes found: {df['problem_class'].unique()}")
print(f"📊 Class distribution:")
print(df['problem_class'].value_counts())

# Combine all text fields
df['combined_text'] = (
    df['title'] + ' ' + 
    df['description'] + ' ' + 
    df['input_description'] + ' ' + 
    df['output_description']
)

print(f"\nCombined text created for {len(df)} problems")

# ============================================
# STEP 3: FEATURE ENGINEERING
# ============================================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

def extract_features(text):
    """Extract useful features from text"""
    features = {}
    
    # Text length
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Mathematical symbols count
    math_symbols = r'[\+\-\*/\=\<\>\(\)\[\]\{\}]'
    features['math_symbols'] = len(re.findall(math_symbols, text))
    
    # Algorithm keywords
    keywords = {
        'graph': ['graph', 'node', 'edge', 'tree', 'path', 'vertex'],
        'dp': ['dynamic', 'dp', 'memoization', 'optimal substructure'],
        'greedy': ['greedy', 'optimal', 'maximize', 'minimize'],
        'sorting': ['sort', 'sorted', 'order', 'arrange'],
        'array': ['array', 'list', 'sequence'],
        'string': ['string', 'character', 'substring'],
        'recursion': ['recursive', 'recursion', 'backtrack'],
        'binary': ['binary', 'bit', 'bitwise'],
        'math': ['prime', 'factorial', 'modulo', 'gcd', 'lcm']
    }
    
    text_lower = text.lower()
    for category, words in keywords.items():
        features[f'has_{category}'] = any(word in text_lower for word in words)
    
    # Number counts
    features['number_count'] = len(re.findall(r'\d+', text))
    
    return features

# Extract features for all problems
print("Extracting features...")
feature_dicts = df['combined_text'].apply(extract_features)
feature_df = pd.DataFrame(feature_dicts.tolist())

print(f"Extracted {len(feature_df.columns)} handcrafted features")
print(f"Features: {feature_df.columns.tolist()}")

# ============================================
# STEP 4: TF-IDF VECTORIZATION
# ============================================
print("\n" + "="*50)
print("TF-IDF VECTORIZATION")
print("="*50)

tfidf = TfidfVectorizer(
    max_features=500,  # Limit to top 500 features
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8  # Ignore terms that appear in more than 80% of documents
)

print("Fitting TF-IDF vectorizer...")
tfidf_features = tfidf.fit_transform(df['combined_text'])
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
)

print(f"TF-IDF features shape: {tfidf_df.shape}")

# Combine all features
X = pd.concat([feature_df, tfidf_df], axis=1)
print(f"\nTotal features: {X.shape[1]}")

# ============================================
# STEP 5: PREPARE TARGETS
# ============================================
y_class = df['problem_class']  # For classification
y_score = df['problem_score']   # For regression

print(f"\nClass distribution:")
print(y_class.value_counts())
print(f"\nScore statistics:")
print(y_score.describe())

# ============================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================
print("\n" + "="*50)
print("SPLITTING DATA")
print("="*50)

X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================
# STEP 7: CLASSIFICATION MODEL
# ============================================
print("\n" + "="*50)
print("TRAINING CLASSIFICATION MODEL")
print("="*50)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("Training classifier...")
clf.fit(X_train, y_class_train)

# Predictions
y_class_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_class_test, y_class_pred)
print(f"\nClassification Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_class_test, y_class_pred))

# ============================================
# STEP 8: REGRESSION MODEL
# ============================================
print("\n" + "="*50)
print("TRAINING REGRESSION MODEL")
print("="*50)

reg = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

print("Training regressor...")
reg.fit(X_train, y_score_train)

# Predictions
y_score_pred = reg.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_score_test, y_score_pred)
rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
print(f"\nRegression MAE: {mae:.4f}")
print(f"Regression RMSE: {rmse:.4f}")

# ============================================
# STEP 9: SAVE MODELS AND VECTORIZER
# ============================================
print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

# Save TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Saved TF-IDF vectorizer")

# Save classification model
with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("✓ Saved classification model")

# Save regression model
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(reg, f)
print("✓ Saved regression model")

# Save feature names for consistency
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("✓ Saved feature names")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nGenerated files:")
print("  - tfidf_vectorizer.pkl")
print("  - classifier_model.pkl")
print("  - regression_model.pkl")
print("  - feature_names.pkl")
print("\nYou can now run the web application using app.py")