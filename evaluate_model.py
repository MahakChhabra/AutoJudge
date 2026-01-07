import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("MODEL EVALUATION & ACCURACY ANALYSIS")
print("="*70)

# ============================================
# LOAD MODELS AND DATA
# ============================================
print("\n📂 Loading models and data...")

# Load models
with open('classifier_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('regression_model.pkl', 'rb') as f:
    regressor = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✅ Models loaded successfully!")

# Load dataset
print("\n📊 Loading dataset...")
data_list = []
with open('programming_problems.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))

df = pd.DataFrame(data_list)
print(f"✅ Loaded {len(df)} problems")

# ============================================
# DATA BALANCING (Same as train_advanced)
# ============================================
print("\n" + "="*70)
print("⚖️ BALANCING DATASET (Same as Training)")
print("="*70)

print("\n📊 Original distribution:")
print(df['problem_class'].value_counts())

# Separate by class
df_hard = df[df['problem_class'] == 'hard']
df_medium = df[df['problem_class'] == 'medium']
df_easy = df[df['problem_class'] == 'easy']

# Calculate target size
target_size = int((len(df_hard) + len(df_medium) + len(df_easy)) / 3)

# Balance
df_hard_downsampled = resample(df_hard, n_samples=target_size, random_state=42)
df_medium_sampled = resample(df_medium, n_samples=target_size, random_state=42, replace=len(df_medium) < target_size)
df_easy_upsampled = resample(df_easy, n_samples=target_size, random_state=42, replace=True)

df_balanced = pd.concat([df_hard_downsampled, df_medium_sampled, df_easy_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 Balanced distribution:")
print(df_balanced['problem_class'].value_counts())

# Use balanced dataset
df = df_balanced

# ============================================
# PREPARE DATA (Same as train_advanced)
# ============================================
print("\n🔧 Preparing features...")

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

# Extract features (same function as train_advanced)
import re

def extract_advanced_features(text):
    features = {}
    words = text.split()
    features['text_length'] = len(text)
    features['word_count'] = len(words)
    features['unique_words'] = len(set(words))
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['sentence_count'] = len(text.split('.'))
    features['paragraph_count'] = len(text.split('\n\n'))
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
    for category, keywords in algorithm_keywords.items():
        count = sum(text_lower.count(keyword) for keyword in keywords)
        features[f'count_{category}'] = count
        features[f'has_{category}'] = count > 0
    
    features['mentions_test_cases'] = 'test case' in text_lower
    features['mentions_constraints'] = 'constraint' in text_lower or 'note' in text_lower
    features['mentions_time_complexity'] = 'time complexity' in text_lower or 'time limit' in text_lower
    features['mentions_space_complexity'] = 'space complexity' in text_lower or 'memory limit' in text_lower
    features['has_multiple_queries'] = text_lower.count('quer') > 2
    features['has_nested_loops_hint'] = 'nested' in text_lower
    features['multiple_test_cases'] = 't test case' in text_lower or 't queries' in text_lower
    features['has_2d_array'] = 'matrix' in text_lower or 'grid' in text_lower or '2d array' in text_lower
    features['has_graph_input'] = 'edge' in text_lower and 'node' in text_lower
    features['optimization_problem'] = 'minimize' in text_lower or 'maximize' in text_lower or 'optimal' in text_lower
    features['counting_problem'] = 'count' in text_lower or 'number of' in text_lower or 'how many' in text_lower
    features['decision_problem'] = 'possible' in text_lower or 'can you' in text_lower or 'determine if' in text_lower
    
    return features

feature_dicts = df['combined_text'].apply(extract_advanced_features)
feature_df = pd.DataFrame(feature_dicts.tolist())

# TF-IDF
tfidf_features = tfidf.transform(df['combined_text'])
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
)

X = pd.concat([feature_df, tfidf_df], axis=1)

# Replace any infinite values with large finite numbers
X = X.replace([np.inf, -np.inf], 10**9)

# Fill any NaN values with 0
X = X.fillna(0)

y_class = df['problem_class']
y_score = df['problem_score']

print(f"✅ Features prepared: {X.shape[1]} features")

# ============================================
# TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

# ============================================
# CLASSIFICATION EVALUATION
# ============================================
print("\n" + "="*70)
print("📊 CLASSIFICATION MODEL EVALUATION")
print("="*70)

y_class_pred = classifier.predict(X_test)

# Metrics
accuracy = accuracy_score(y_class_test, y_class_pred)
precision = precision_score(y_class_test, y_class_pred, average='weighted')
recall = recall_score(y_class_test, y_class_pred, average='weighted')
f1 = f1_score(y_class_test, y_class_pred, average='weighted')

print(f"\n📈 Overall Metrics:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

print(f"\n📋 Per-Class Performance:")
print(classification_report(y_class_test, y_class_pred))

# Confusion Matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_class_test, y_class_pred)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
labels = sorted(y_class.unique())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, 
            yticklabels=labels)
plt.title('Confusion Matrix - Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.close()

# ============================================
# CROSS-VALIDATION
# ============================================
print("\n" + "="*70)
print("🔄 CROSS-VALIDATION (5-Fold)")
print("="*70)

try:
    cv_scores = cross_val_score(classifier, X, y_class, cv=5, scoring='accuracy')
    print(f"\n📊 CV Scores: {cv_scores}")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
except:
    print("⚠️  Cross-validation skipped (ensemble models may take long)")

# ============================================
# REGRESSION EVALUATION
# ============================================
print("\n" + "="*70)
print("📊 REGRESSION MODEL EVALUATION")
print("="*70)

y_score_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_score_test, y_score_pred)
rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
r2 = r2_score(y_score_test, y_score_pred)

print(f"\n📈 Regression Metrics:")
print(f"   MAE (Mean Absolute Error):  {mae:.4f}")
print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"   R² Score: {r2:.4f}")

# Plot Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_score_test, y_score_pred, alpha=0.5)
plt.plot([y_score_test.min(), y_score_test.max()], 
         [y_score_test.min(), y_score_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Score', fontsize=12)
plt.ylabel('Predicted Score', fontsize=12)
plt.title('Regression: Predicted vs Actual Scores', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: regression_plot.png")
plt.close()

# ============================================
# ERROR ANALYSIS
# ============================================
print("\n" + "="*70)
print("🔍 ERROR ANALYSIS")
print("="*70)

errors = X_test[y_class_test != y_class_pred]
print(f"\n❌ Misclassified: {len(errors)} out of {len(X_test)} ({len(errors)/len(X_test)*100:.2f}%)")

print("\n🔀 Most Common Misclassifications:")
error_df = pd.DataFrame({
    'true': y_class_test[y_class_test != y_class_pred],
    'predicted': y_class_pred[y_class_test != y_class_pred]
})
if len(error_df) > 0:
    error_counts = error_df.groupby(['true', 'predicted']).size().sort_values(ascending=False)
    print(error_counts.head(10))

# ============================================
# FEATURE IMPORTANCE (if available)
# ============================================
print("\n" + "="*70)
print("🎯 FEATURE IMPORTANCE")
print("="*70)

try:
    if hasattr(classifier, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': classifier.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(classifier, 'estimators_'):
        # For ensemble, average importance
        importances = []
        for est in classifier.estimators_:
            if hasattr(est, 'feature_importances_'):
                importances.append(est.feature_importances_)
        if importances:
            avg_importance = np.mean(importances, axis=0)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features:")
    print(feature_importance.head(20))
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: feature_importance.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Feature importance not available for this model type")

# ============================================
# PERFORMANCE BY CLASS
# ============================================
print("\n" + "="*70)
print("📊 PER-CLASS ACCURACY BREAKDOWN")
print("="*70)

for cls in sorted(y_class.unique()):
    mask = y_class_test == cls
    cls_acc = accuracy_score(y_class_test[mask], y_class_pred[mask])
    cls_count = mask.sum()
    print(f"\n{cls.upper()}:")
    print(f"   Samples: {cls_count}")
    print(f"   Accuracy: {cls_acc:.4f} ({cls_acc*100:.2f}%)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print(f"\n   📈 Classification Accuracy: {accuracy*100:.2f}%")
print(f"   📉 Regression MAE: {mae:.2f}")
print(f"   📉 Regression RMSE: {rmse:.2f}")
print(f"   📈 R² Score: {r2:.4f}")
print(f"\n   📁 Generated visualizations:")
print(f"      - confusion_matrix.png")
print(f"      - regression_plot.png")
print(f"      - feature_importance.png")

# Interpretation
print(f"\n   💡 Interpretation:")
if accuracy > 0.85:
    print(f"      🎉 EXCELLENT! Model is performing very well!")
elif accuracy > 0.75:
    print(f"      ✅ GOOD! Model is performing well.")
elif accuracy > 0.65:
    print(f"      ⚠️  FAIR. Consider collecting more data or tuning.")
else:
    print(f"      ❌ NEEDS IMPROVEMENT. Check data quality and features.")

print("\n" + "="*70)