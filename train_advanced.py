import pandas as pd
import numpy as np
import re
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 ADVANCED MODEL TRAINING - MAXIMUM ACCURACY")
print("="*70)

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print("\n📂 Loading dataset...")
data_list = []
with open('programming_problems.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))

df = pd.DataFrame(data_list)
print(f"✅ Loaded {len(df)} problems")

# ============================================
# STEP 2: DATA BALANCING (CRITICAL!)
# ============================================
print("\n" + "="*70)
print("⚖️ BALANCING DATASET")
print("="*70)

print("\n📊 Original distribution:")
print(df['problem_class'].value_counts())

# Separate by class
df_hard = df[df['problem_class'] == 'hard']
df_medium = df[df['problem_class'] == 'medium']
df_easy = df[df['problem_class'] == 'easy']

# Calculate target size (use average)
target_size = int((len(df_hard) + len(df_medium) + len(df_easy)) / 3)
print(f"\n🎯 Target size per class: {target_size}")

# Downsample hard, upsample easy
df_hard_downsampled = resample(df_hard, n_samples=target_size, random_state=42)
df_medium_sampled = resample(df_medium, n_samples=target_size, random_state=42, replace=len(df_medium) < target_size)
df_easy_upsampled = resample(df_easy, n_samples=target_size, random_state=42, replace=True)

# Combine
df_balanced = pd.concat([df_hard_downsampled, df_medium_sampled, df_easy_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 Balanced distribution:")
print(df_balanced['problem_class'].value_counts())
print(f"\n✅ Total samples after balancing: {len(df_balanced)}")

# Use balanced dataset
df = df_balanced

# ============================================
# STEP 3: DATA PREPROCESSING
# ============================================
print("\n" + "="*70)
print("🔧 PREPROCESSING DATA")
print("="*70)

df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')
df['input_description'] = df['input_description'].fillna('')
df['output_description'] = df['output_description'].fillna('')

df['combined_text'] = (
    df['title'] + ' ' + 
    df['description'] + ' ' + 
    df['input_description'] + ' ' + 
    df['output_description']
)

# ============================================
# STEP 4: ADVANCED FEATURE ENGINEERING
# ============================================
print("\n" + "="*70)
print("🎯 ADVANCED FEATURE ENGINEERING")
print("="*70)

def extract_advanced_features(text):
    """Extract comprehensive features"""
    features = {}
    
    # Basic text statistics
    words = text.split()
    features['text_length'] = len(text)
    features['word_count'] = len(words)
    features['unique_words'] = len(set(words))
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['sentence_count'] = len(text.split('.'))
    features['paragraph_count'] = len(text.split('\n\n'))
    
    # Complexity indicators
    features['math_symbols'] = len(re.findall(r'[\+\-\*/\=\<\>\(\)\[\]\{\}]', text))
    features['number_count'] = len(re.findall(r'\d+', text))
    
    # Extract large numbers (constraints indicator) - with safe handling
    numbers = [int(n) for n in re.findall(r'\d+', text) if n.isdigit() and len(n) < 15]  # Limit number length
    if numbers:
        max_num = max(numbers)
        features['max_number'] = min(max_num, 10**9)  # Cap at 1 billion
        features['has_large_constraint'] = max_num > 100000
        features['has_huge_constraint'] = max_num > 1000000
    else:
        features['max_number'] = 0
        features['has_large_constraint'] = False
        features['has_huge_constraint'] = False
    
    # Advanced algorithm keywords with scoring
    algorithm_keywords = {
        'graph_advanced': ['dijkstra', 'bellman', 'floyd', 'kruskal', 'prim', 'topological', 'strongly connected', 'bipartite'],
        'graph_basic': ['graph', 'node', 'edge', 'tree', 'dfs', 'bfs', 'path', 'cycle'],
        'dp_advanced': ['knapsack', 'longest common', 'edit distance', 'matrix chain', 'optimal substructure'],
        'dp_basic': ['dynamic', 'dp', 'memoization', 'subproblem'],
        'greedy': ['greedy', 'activity selection', 'huffman', 'fractional'],
        'sorting': ['sort', 'merge sort', 'quick sort', 'heap sort', 'counting sort'],
        'searching': ['binary search', 'ternary search', 'exponential search'],
        'data_structures': ['stack', 'queue', 'heap', 'priority queue', 'trie', 'segment tree', 'fenwick'],
        'string_advanced': ['kmp', 'rabin karp', 'suffix', 'z algorithm', 'manacher'],
        'string_basic': ['string', 'substring', 'pattern', 'palindrome'],
        'math_advanced': ['number theory', 'modular arithmetic', 'chinese remainder', 'fermat', 'euler'],
        'math_basic': ['prime', 'factorial', 'gcd', 'lcm', 'fibonacci', 'modulo'],
        'geometry': ['geometry', 'coordinate', 'distance', 'convex hull', 'line intersection'],
        'bit_manipulation': ['bitwise', 'xor', 'bit mask', 'bit manipulation'],
        'recursion': ['recursive', 'backtrack', 'divide and conquer'],
        'two_pointer': ['two pointer', 'sliding window'],
        'simulation': ['simulate', 'simulation']
    }
    
    text_lower = text.lower()
    
    # Count occurrences for each category
    for category, keywords in algorithm_keywords.items():
        count = sum(text_lower.count(keyword) for keyword in keywords)
        features[f'count_{category}'] = count
        features[f'has_{category}'] = count > 0
    
    # Difficulty indicators
    features['mentions_test_cases'] = 'test case' in text_lower
    features['mentions_constraints'] = 'constraint' in text_lower or 'note' in text_lower
    features['mentions_time_complexity'] = 'time complexity' in text_lower or 'time limit' in text_lower
    features['mentions_space_complexity'] = 'space complexity' in text_lower or 'memory limit' in text_lower
    features['has_multiple_queries'] = text_lower.count('quer') > 2
    features['has_nested_loops_hint'] = 'nested' in text_lower
    
    # Input/Output complexity
    features['multiple_test_cases'] = 't test case' in text_lower or 't queries' in text_lower
    features['has_2d_array'] = 'matrix' in text_lower or 'grid' in text_lower or '2d array' in text_lower
    features['has_graph_input'] = 'edge' in text_lower and 'node' in text_lower
    
    # Problem type hints
    features['optimization_problem'] = 'minimize' in text_lower or 'maximize' in text_lower or 'optimal' in text_lower
    features['counting_problem'] = 'count' in text_lower or 'number of' in text_lower or 'how many' in text_lower
    features['decision_problem'] = 'possible' in text_lower or 'can you' in text_lower or 'determine if' in text_lower
    
    return features

print("Extracting advanced features...")
feature_dicts = df['combined_text'].apply(extract_advanced_features)
feature_df = pd.DataFrame(feature_dicts.tolist())

print(f"✅ Extracted {len(feature_df.columns)} handcrafted features")

# ============================================
# STEP 5: ENHANCED TF-IDF
# ============================================
print("\n📊 Creating enhanced TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=2000,  # Increased to 2000
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    strip_accents='unicode',
    lowercase=True
)

tfidf_features = tfidf.fit_transform(df['combined_text'])
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
)

print(f"✅ Created {tfidf_df.shape[1]} TF-IDF features")

# Combine features
X = pd.concat([feature_df, tfidf_df], axis=1)

# Replace any infinite values with large finite numbers
X = X.replace([np.inf, -np.inf], 10**9)

# Fill any NaN values with 0
X = X.fillna(0)

y_class = df['problem_class']
y_score = df['problem_score']

# Encode class labels for models that need numeric labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)

print(f"\n📈 Total features: {X.shape[1]}")
print(f"✅ Cleaned data: no infinities or NaN values")
print(f"✅ Class encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# ============================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_class_train, y_class_test, y_class_train_enc, y_class_test_enc, y_score_train, y_score_test = train_test_split(
    X, y_class, y_class_encoded, y_score, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\n✅ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")

# ============================================
# STEP 7: ENSEMBLE CLASSIFICATION
# ============================================
print("\n" + "="*70)
print("🤖 TRAINING ENSEMBLE CLASSIFIER")
print("="*70)

# Create multiple models
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

# Try individual models first
models = {
    'Random Forest': rf,
    'XGBoost': xgb,
    'LightGBM': lgbm
}

best_model = None
best_acc = 0
best_name = ""

print("\n🔄 Training individual models...")
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    
    # Use encoded labels for XGBoost and LightGBM, original for Random Forest
    if 'XG' in name or 'Light' in name:
        model.fit(X_train, y_class_train_enc)
        y_pred_encoded = model.predict(X_test)
        # Convert back to original labels
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        model.fit(X_train, y_class_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_class_test, y_pred)
    
    print(f"✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# Create ensemble
print("\n" + "="*50)
print("🎯 Creating Ensemble Model")
print("="*50)

# Re-train models for ensemble
rf_ensemble = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=3, random_state=42, n_jobs=-1)
xgb_ensemble = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
lgbm_ensemble = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)

# Train with appropriate labels
rf_ensemble.fit(X_train, y_class_train)
xgb_ensemble.fit(X_train, y_class_train_enc)
lgbm_ensemble.fit(X_train, y_class_train_enc)

# Create a custom ensemble predictor
class CustomEnsemble:
    def __init__(self, rf, xgb, lgbm, label_encoder):
        self.rf = rf
        self.xgb = xgb
        self.lgbm = lgbm
        self.label_encoder = label_encoder
    
    def predict(self, X):
        pred_rf = self.rf.predict(X)
        pred_xgb = self.label_encoder.inverse_transform(self.xgb.predict(X))
        pred_lgbm = self.label_encoder.inverse_transform(self.lgbm.predict(X))
        
        # Majority voting
        predictions = np.array([pred_rf, pred_xgb, pred_lgbm])
        final_pred = []
        for i in range(len(X)):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_pred.append(unique[np.argmax(counts)])
        return np.array(final_pred)
    
    def predict_proba(self, X):
        # Average probabilities
        prob_rf = self.rf.predict_proba(X)
        prob_xgb = self.xgb.predict_proba(X)
        prob_lgbm = self.lgbm.predict_proba(X)
        return (prob_rf + prob_xgb + prob_lgbm) / 3

ensemble = CustomEnsemble(rf_ensemble, xgb_ensemble, lgbm_ensemble, label_encoder)

print("Training ensemble...")
y_pred_ensemble = ensemble.predict(X_test)
acc_ensemble = accuracy_score(y_class_test, y_pred_ensemble)

print(f"\n🎉 Ensemble Accuracy: {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")

# Choose best between individual and ensemble
if acc_ensemble > best_acc:
    final_classifier = ensemble
    final_acc = acc_ensemble
    final_name = "Ensemble (RF + XGB + LGBM)"
else:
    final_classifier = best_model
    final_acc = best_acc
    final_name = best_name

print("\n" + "="*70)
print(f"🏆 BEST CLASSIFIER: {final_name}")
print(f"🎯 Final Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
print("="*70)

y_class_pred = final_classifier.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_class_test, y_class_pred))

# ============================================
# STEP 8: REGRESSION MODEL
# ============================================
print("\n" + "="*70)
print("📈 TRAINING REGRESSION MODEL")
print("="*70)

reg_xgb = XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)

reg_lgbm = LGBMRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

print("Training XGBoost regressor...")
reg_xgb.fit(X_train, y_score_train)
y_pred_xgb = reg_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_score_test, y_pred_xgb)

print("Training LightGBM regressor...")
reg_lgbm.fit(X_train, y_score_train)
y_pred_lgbm = reg_lgbm.predict(X_test)
mae_lgbm = mean_absolute_error(y_score_test, y_pred_lgbm)

print(f"\nXGBoost MAE: {mae_xgb:.4f}")
print(f"LightGBM MAE: {mae_lgbm:.4f}")

# Choose best
if mae_lgbm < mae_xgb:
    final_regressor = reg_lgbm
    final_mae = mae_lgbm
    print(f"\n🏆 Best Regressor: LightGBM")
else:
    final_regressor = reg_xgb
    final_mae = mae_xgb
    print(f"\n🏆 Best Regressor: XGBoost")

print(f"🎯 Final MAE: {final_mae:.4f}")

# ============================================
# STEP 9: SAVE MODELS
# ============================================
print("\n" + "="*70)
print("💾 SAVING MODELS")
print("="*70)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(final_classifier, f)

with open('regression_model.pkl', 'wb') as f:
    pickle.dump(final_regressor, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Save label encoder for models that need it
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ All models saved!")

print("\n" + "="*70)
print("✨ TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Final Results:")
print(f"   Classification Accuracy: {final_acc*100:.2f}%")
print(f"   Regression MAE: {final_mae:.4f}")
print(f"\n💡 Models saved with advanced features!")
print("="*70)