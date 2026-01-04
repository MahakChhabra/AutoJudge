from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import re
import numpy as np

app = Flask(__name__)

# ============================================
# LOAD MODELS AND VECTORIZER
# ============================================
print("Loading models...")

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('classifier_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('regression_model.pkl', 'rb') as f:
    regressor = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✓ Models loaded successfully!")

# ============================================
# FEATURE EXTRACTION FUNCTION
# ============================================
def extract_features(text):
    """Extract handcrafted features from text"""
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

def prepare_features(title, description, input_desc, output_desc):
    """Prepare features for prediction"""
    # Combine text
    combined_text = f"{title} {description} {input_desc} {output_desc}"
    
    # Extract handcrafted features
    handcrafted = extract_features(combined_text)
    
    # Get TF-IDF features
    tfidf_features = tfidf.transform([combined_text])
    tfidf_dict = {f'tfidf_{i}': val for i, val in enumerate(tfidf_features.toarray()[0])}
    
    # Combine all features
    all_features = {**handcrafted, **tfidf_dict}
    
    # Create DataFrame with correct column order
    feature_df = pd.DataFrame([all_features])
    
    # Ensure all required features are present (fill missing with 0)
    for col in feature_names:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    # Reorder columns to match training
    feature_df = feature_df[feature_names]
    
    return feature_df

# ============================================
# ROUTES
# ============================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        title = data.get('title', '')
        description = data.get('description', '')
        input_desc = data.get('input_description', '')
        output_desc = data.get('output_description', '')
        
        # Prepare features
        features = prepare_features(title, description, input_desc, output_desc)
        
        # Make predictions
        predicted_class = classifier.predict(features)[0]
        predicted_score = regressor.predict(features)[0]
        
        # Get prediction probabilities for classification
        class_probs = classifier.predict_proba(features)[0]
        class_names = classifier.classes_
        
        confidence = {class_names[i]: float(class_probs[i]) for i in range(len(class_names))}
        
        # Return results
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'predicted_score': round(float(predicted_score), 2),
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ============================================
# RUN APP
# ============================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("AutoJudge Web Application")
    print("="*50)
    print("Server starting...")
    print("Access the application at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)