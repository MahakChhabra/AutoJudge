# 📊 Complete Guide: Check & Improve Model Accuracy

## 🎯 How to Check Accuracy

### Method 1: Quick Check (Already in training output)
When you run `train_model.py`, you'll see:
```
Classification Accuracy: 0.XXXX
Regression MAE: X.XXXX
Regression RMSE: X.XXXX
```

### Method 2: Detailed Evaluation
Run the evaluation script I created:

```bash
python evaluate_model.py
```

This will show you:
- ✅ **Overall Accuracy** - How often the model is correct
- ✅ **Precision** - Of all predicted Easy/Medium/Hard, how many were correct
- ✅ **Recall** - Of all actual Easy/Medium/Hard, how many were found
- ✅ **F1-Score** - Harmonic mean of precision and recall
- ✅ **Confusion Matrix** - Visual representation of predictions vs actual
- ✅ **Per-class Performance** - Individual accuracy for Easy, Medium, Hard
- ✅ **Feature Importance** - Which features matter most
- ✅ **Error Analysis** - Which problems are being misclassified

### Output Files Generated:
- `confusion_matrix.png` - Visual confusion matrix
- `regression_plot.png` - Predicted vs actual scores
- `feature_importance.png` - Most important features

---

## 📈 Understanding Accuracy Metrics

### Classification Metrics:

**1. Accuracy**
- Overall correctness: (Correct Predictions) / (Total Predictions)
- **Good**: > 85%
- **Acceptable**: 70-85%
- **Needs Improvement**: < 70%

**2. Precision**
- Of all predictions for a class, how many were correct
- High precision = Few false positives

**3. Recall**
- Of all actual instances of a class, how many were found
- High recall = Few false negatives

**4. F1-Score**
- Balance between precision and recall
- Best when both are high

### Regression Metrics:

**1. MAE (Mean Absolute Error)**
- Average difference between predicted and actual scores
- Lower is better
- **Good**: < 100 (for score range 800-3000)
- **Acceptable**: 100-200
- **Needs Improvement**: > 200

**2. RMSE (Root Mean Squared Error)**
- Like MAE but penalizes large errors more
- Lower is better

**3. R² Score**
- How well predictions fit the data (0 to 1)
- **Good**: > 0.8
- **Acceptable**: 0.6-0.8
- **Needs Improvement**: < 0.6

---

## 🚀 How to Improve Accuracy

### Level 1: Quick Improvements (Easiest)

#### 1. Use the Improved Training Script
```bash
python train_improved.py
```

This script includes:
- ✅ More features (50+ instead of 10)
- ✅ Better TF-IDF (1000 features, trigrams)
- ✅ Hyperparameter tuning
- ✅ Multiple model comparison (Random Forest vs XGBoost)
- ✅ Enhanced keyword detection

**Expected improvement**: +5-10% accuracy

#### 2. Install XGBoost (if not installed)
```bash
pip install xgboost
```

XGBoost often performs better than Random Forest.

---

### Level 2: Data Improvements (Medium Difficulty)

#### 1. Collect More Data
- **Current**: You need at least 500-1000 problems
- **Ideal**: 2000+ problems
- More data = Better learning

#### 2. Balance Your Dataset
Check class distribution:
```python
print(df['problem_class'].value_counts())
```

If imbalanced (e.g., 1000 Easy, 100 Hard):
```python
from sklearn.utils import resample

# Oversample minority classes
df_easy = df[df['problem_class'] == 'Easy']
df_medium = df[df['problem_class'] == 'Medium']
df_hard = df[df['problem_class'] == 'Hard']

# Upsample Hard and Medium
df_medium_upsampled = resample(df_medium, n_samples=len(df_easy), random_state=42)
df_hard_upsampled = resample(df_hard, n_samples=len(df_easy), random_state=42)

# Combine
df_balanced = pd.concat([df_easy, df_medium_upsampled, df_hard_upsampled])
```

#### 3. Clean Your Data
- Remove duplicates
- Fix inconsistent labels
- Handle missing values properly
- Remove low-quality problems

---

### Level 3: Advanced Feature Engineering (Advanced)

#### 1. Add Word Embeddings
Instead of just TF-IDF, use pre-trained embeddings:

```python
# Using sentence transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['combined_text'].tolist())
```

#### 2. Add Code-Specific Features
If problems contain code examples:
```python
def extract_code_features(text):
    return {
        'has_code_block': '```' in text or 'code' in text.lower(),
        'has_example': 'example' in text.lower(),
        'loop_indicators': text.lower().count('for') + text.lower().count('while'),
        'conditional_indicators': text.lower().count('if')
    }
```

#### 3. Extract Complexity Hints
```python
def extract_complexity_features(text):
    text_lower = text.lower()
    return {
        'mentions_o_n': 'o(n)' in text_lower or 'o(n^2)' in text_lower,
        'mentions_log': 'log' in text_lower,
        'mentions_exponential': 'exponential' in text_lower,
        'has_constraints_million': any(int(x) > 1000000 for x in re.findall(r'\d+', text))
    }
```

---

### Level 4: Advanced Models (Expert)

#### 1. Try Deep Learning Models

**Option A: Simple Neural Network**
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

**Option B: BERT (Best but requires more resources)**
```python
from transformers import BertTokenizer, TFBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
```

#### 2. Ensemble Methods
Combine multiple models:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier()),
    ('svc', SVC(probability=True))
], voting='soft')
```

---

## 🎯 Step-by-Step Improvement Plan

### Week 1: Quick Wins
1. ✅ Run `evaluate_model.py` to check current performance
2. ✅ Run `train_improved.py` with enhanced features
3. ✅ Install XGBoost
4. ✅ Check for data quality issues

**Expected improvement**: +10-15%

### Week 2: Data Improvements
1. ✅ Collect more training data
2. ✅ Balance class distribution
3. ✅ Clean and deduplicate data
4. ✅ Add more domain keywords

**Expected improvement**: +5-10%

### Week 3: Advanced Features
1. ✅ Add word embeddings
2. ✅ Extract complexity hints
3. ✅ Add code-specific features
4. ✅ Try ensemble methods

**Expected improvement**: +5-10%

### Week 4: Deep Learning (Optional)
1. ✅ Try BERT or similar transformers
2. ✅ Fine-tune on your dataset
3. ✅ Compare with traditional methods

**Expected improvement**: +10-20%

---

## 📊 Benchmarks

| Approach | Expected Accuracy | Difficulty | Time |
|----------|------------------|------------|------|
| Basic (Original) | 65-75% | Easy | 5 min |
| Improved Features | 75-85% | Easy | 10 min |
| XGBoost + Tuning | 80-90% | Medium | 30 min |
| Balanced Data | +5-10% | Medium | 1 hour |
| Word Embeddings | 85-92% | Advanced | 2 hours |
| BERT/Transformers | 90-95% | Expert | 4+ hours |

---

## 🔍 Debugging Low Accuracy

### If accuracy is < 60%:

**Check 1: Data Quality**
```python
# Look for issues
print(df.isnull().sum())  # Missing values?
print(df['problem_class'].value_counts())  # Imbalanced?
print(df['combined_text'].str.len().describe())  # Text too short?
```

**Check 2: Feature Issues**
```python
# Check if features are being extracted
print(X.head())
print(X.describe())
```

**Check 3: Labels Consistency**
```python
# Are labels consistent?
print(df.groupby('problem_class')['problem_score'].describe())
```

### Common Issues:

1. **Too little data** → Collect more (need 500+ minimum)
2. **Imbalanced classes** → Use resampling or class weights
3. **Poor text quality** → Clean and preprocess better
4. **Wrong features** → Add domain-specific features
5. **Model overfitting** → Use cross-validation, reduce complexity

---

## 💡 Pro Tips

1. **Always use cross-validation** - Single train/test split can be misleading
2. **Check feature importance** - Remove useless features
3. **Look at misclassifications** - Learn from errors
4. **Start simple, add complexity** - Don't jump to deep learning immediately
5. **Track improvements** - Keep a log of what works
6. **Use class weights** for imbalanced data:
   ```python
   clf = RandomForestClassifier(class_weight='balanced')
   ```

---

## 📝 Quick Checklist

Before trying complex methods, ensure:

- [ ] Dataset has 500+ problems
- [ ] Classes are somewhat balanced (max 3:1 ratio)
- [ ] Text quality is good (no garbled text)
- [ ] Features are being extracted correctly
- [ ] Model is not overfitting (train acc >> test acc)
- [ ] Using appropriate evaluation metrics
- [ ] Tried basic hyperparameter tuning

---

## 🎓 Summary

**To check accuracy:**
```bash
python evaluate_model.py
```

**To improve accuracy (in order):**
1. Run `train_improved.py` (easiest, +10%)
2. Get more data (medium, +10%)
3. Balance classes (medium, +5%)
4. Add embeddings (hard, +10%)
5. Try deep learning (expert, +15%)

**Start with the improved training script - it's the easiest and gives good results!**