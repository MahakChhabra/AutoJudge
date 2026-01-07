import pandas as pd
import numpy as np
import re
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMPROVED MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("="*70)

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print("\n📂 Loading dataset from JSONL...")
data_list = []
with open('programming_problems.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))

df = pd.DataFrame(data_list)
print(f"✅ Loaded {len(df)} problems")

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("\n🔧 Preprocessing data...")
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')
df['input_description'] = df['input_description'].fillna('')
df['output_description'] = df['output_description'].fillna('')

# Keep original class labels (lowercase)
print(f"\n📊 Classes found: {df['problem_class'].unique()}")
print(f"📊 Class distribution:")
print(df['problem_class'].value_counts())

df['combined_text'] = (
    df['title'] + ' ' + 
    df['description'] + ' ' + 
    df['input_description'] + ' ' + 
    df['output_description']
)

# ============================================
# STEP 3: ENHANCED FEATURE ENGINEERING
# ============================================
print("\n🎯 Enhanced feature engineering...")

def extract_enhanced_features(text):
    """Extract enhanced features with more domain knowledge"""
    features = {}
    
    # Basic text stats
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0
    features['sentence_count'] = len(text.split('.'))
    
    # Mathematical symbols
    math_symbols = r'[\+\-\*/\=\<\>\(\)\[\]\{\}]'
    features['math_symbols'] = len(re.findall(math_symbols, text))
    
    # Enhanced algorithm keywords with more terms
    keywords = {
        'graph': ['graph', 'node', 'edge', 'tree', 'path', 'vertex', 'dfs', 'bfs', 
                  'dijkstra', 'shortest path', 'cycle', 'connected'],
        'dp': ['dynamic', 'dp', 'memoization', 'optimal substructure', 'overlapping',
               'knapsack', 'longest common', 'edit distance'],
        'greedy': ['greedy', 'optimal', 'maximize', 'minimize', 'local optimal'],
        'sorting': ['sort', 'sorted', 'order', 'arrange', 'merge sort', 'quick sort'],
        'array': ['array', 'list', 'sequence', 'subarray', 'contiguous'],
        'string': ['string', 'character', 'substring', 'pattern', 'text'],
        'recursion': ['recursive', 'recursion', 'backtrack', 'divide and conquer'],
        'binary': ['binary', 'bit', 'bitwise', 'xor', 'and', 'or'],
        'math': ['prime', 'factorial', 'modulo', 'gcd', 'lcm', 'fibonacci', 'number theory'],
        'data_structures': ['stack', 'queue', 'heap', 'priority queue', 'hash', 'map'],
        'search': ['binary search', 'linear search', 'search', 'find'],
        'matrix': ['matrix', 'grid', '2d array', 'row', 'column']
    }
    
    text_lower = text.lower()
    for category, words in keywords.items():
        features[f'has_{category}'] = any(word in text_lower for word in words)
        # Count occurrences
        features[f'count_{category}'] = sum(text_lower.count(word) for word in words)
    
    # Number and constraint indicators
    features['number_count'] = len(re.findall(r'\d+', text))
    features['has_large_numbers'] = any(int(num) > 1000000 for num in re.findall(r'\d+', text) if num.isdigit())
    features['has_constraints'] = 'constraints' in text_lower or 'note' in text_lower
    
    # Complexity indicators
    features['has_nested'] = 'nested' in text_lower
    features['has_multiple'] = 'multiple' in text_lower
    features['has_queries'] = 'queries' in text_lower or 'query' in text_lower
    
    # Input/Output complexity
    features['has_test_cases'] = 'test case' in text_lower
    features['mentions_time'] = 'time' in text_lower and 'complexity' in text_lower
    features['mentions_space'] = 'space' in text_lower and 'complexity' in text_lower
    
    return features

print("Extracting enhanced features...")
feature_dicts = df['combined_text'].apply(extract_enhanced_features)
feature_df = pd.DataFrame(feature_dicts.tolist())

print(f"✅ Extracted {len(feature_df.columns)} handcrafted features")

# ============================================
# STEP 4: IMPROVED TF-IDF
# ============================================
print("\n📊 Creating improved TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=1000,  # Increased from 500
    ngram_range=(1, 3),  # Added trigrams
    min_df=2,
    max_df=0.8,
    sublinear_tf=True  # Use sublinear TF scaling
)

tfidf_features = tfidf.fit_transform(df['combined_text'])
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
)

print(f"✅ Created {tfidf_df.shape[1]} TF-IDF features")

# Combine features
X = pd.concat([feature_df, tfidf_df], axis=1)
y_class = df['problem_class']
y_score = df['problem_score']

print(f"\n📈 Total features: {X.shape[1]}")
print(f"📊 Class distribution:\n{y_class.value_counts()}")

# ============================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\n✅ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")

# ============================================
# STEP 6: MODEL SELECTION & HYPERPARAMETER TUNING
# ============================================
print("\n" + "="*70)
print("🤖 TRAINING MULTIPLE MODELS")
print("="*70)

# Try multiple classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', scale_pos_weight=1)
}

best_classifier = None
best_score = 0
best_name = ""

print("\n🔄 Training and comparing classifiers...")

for name, clf in classifiers.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")
    
    if name == 'Random Forest':
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5]
        }
    elif name == 'XGBoost':
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [5, 7, 10],
            'learning_rate': [0.1, 0.01]
        }
    
    # Grid search with limited params for speed
    print(f"Performing hyperparameter tuning...")
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_class_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_class_test, y_pred)
    
    print(f"✅ Best params: {grid_search.best_params_}")
    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy > best_score:
        best_score = accuracy
        best_classifier = best_model
        best_name = name

print("\n" + "="*70)
print(f"🏆 BEST CLASSIFIER: {best_name}")
print(f"🎯 Best Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
print("="*70)

# Final evaluation
y_class_pred = best_classifier.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_class_test, y_class_pred))
print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_class_test, y_class_pred))

# ============================================
# STEP 7: IMPROVED REGRESSION MODEL
# ============================================
print("\n" + "="*70)
print("📈 TRAINING REGRESSION MODEL")
print("="*70)

regressors = {
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

best_regressor = None
best_mae = float('inf')
best_reg_name = ""

for name, reg in regressors.items():
    print(f"\nTraining: {name}")
    
    if name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [5, 7],
            'learning_rate': [0.1, 0.05]
        }
    elif name == 'XGBoost':
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [5, 7],
            'learning_rate': [0.1, 0.05]
        }
    
    grid_search = GridSearchCV(reg, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_score_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_score_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_score_test, y_pred))
    
    print(f"✅ Best params: {grid_search.best_params_}")
    print(f"✅ MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    if mae < best_mae:
        best_mae = mae
        best_regressor = best_model
        best_reg_name = name

print("\n" + "="*70)
print(f"🏆 BEST REGRESSOR: {best_reg_name}")
print(f"🎯 Best MAE: {best_mae:.4f}")
print("="*70)

# ============================================
# STEP 8: SAVE IMPROVED MODELS
# ============================================
print("\n💾 Saving improved models...")

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

with open('regression_model.pkl', 'wb') as f:
    pickle.dump(best_regressor, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ All models saved!")
print("\n" + "="*70)
print("✨ TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Final Results:")
print(f"   Classification Accuracy: {best_score*100:.2f}%")
print(f"   Regression MAE: {best_mae:.4f}")
print("\n💡 Run 'python evaluate_model.py' for detailed analysis!")
print("="*70)