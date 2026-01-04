# AutoJudge: Programming Problem Difficulty Predictor 🤖

An intelligent system that automatically predicts the difficulty class (Easy/Medium/Hard) and difficulty score of programming problems using machine learning.

## 📋 Project Overview

This system analyzes textual descriptions of programming problems and predicts:
- **Problem Class**: Easy / Medium / Hard (Classification)
- **Problem Score**: Numerical difficulty score (Regression)

## 🎯 Features

- Text-based difficulty prediction
- Dual prediction system (classification + regression)
- Interactive web interface
- Feature engineering with TF-IDF and handcrafted features
- Confidence scores for predictions

## 📁 Project Structure

```
autojudge/
│
├── train_model.py                 # Model training script
├── app.py                         # Flask web application
├── inspect_dataset.py             # Dataset inspector tool
├── templates/
│   └── index.html                # Web interface
├── programming_problems.jsonl     # Your dataset (you provide this)
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── classifier_model.pkl          # Saved classification model
├── regression_model.pkl          # Saved regression model
├── feature_names.pkl             # Saved feature names
└── README.md                     # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Install Required Packages

```bash
pip install pandas numpy scikit-learn flask
```

Or create a `requirements.txt` file:

```txt
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
flask==2.3.2
```

Then install:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Requirements

Your dataset JSONL file should have these fields in each JSON object:
- `title` - Problem title
- `description` - Full problem description
- `input_description` - Input format description
- `output_description` - Expected output description
- `problem_class` - Difficulty class (Easy/Medium/Hard)
- `problem_score` - Numerical difficulty score

**Example JSONL format:**
```json
{"title": "Two Sum", "description": "Given an array...", "input_description": "First line...", "output_description": "Two integers...", "problem_class": "Easy", "problem_score": 800}
{"title": "Graph Problem", "description": "Find shortest...", "input_description": "N and M...", "output_description": "Single integer...", "problem_class": "Hard", "problem_score": 2400}
```

**Place your dataset file in the project directory and name it `programming_problems.jsonl`**

## 🚀 Usage

### Step 1: Train the Models

Run the training script:

```bash
python train_model.py
```

This will:
- Load and preprocess your dataset
- Extract features (TF-IDF + handcrafted features)
- Train classification and regression models
- Save models to disk
- Display evaluation metrics

**Expected Output:**
```
Classification Accuracy: X.XXXX
Regression MAE: X.XXXX
Regression RMSE: X.XXXX
```

### Step 2: Run the Web Application

Start the Flask server:

```bash
python app.py
```

### Step 3: Access the Web Interface

Open your browser and go to:
```
http://127.0.0.1:5000
```

### Step 4: Make Predictions

1. Enter problem title
2. Paste problem description
3. Add input description
4. Add output description
5. Click "Predict Difficulty"
6. View predicted class, score, and confidence levels

## 🔍 Feature Engineering

The system extracts multiple types of features:

### Handcrafted Features:
- **Text statistics**: length, word count
- **Mathematical symbols**: count of operators and brackets
- **Algorithm keywords**: graph, DP, greedy, sorting, etc.
- **Number count**: frequency of numerical values

### TF-IDF Features:
- Top 500 most important terms
- Unigrams and bigrams
- Normalized term frequency scores

## 🤖 Models Used

### Classification Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: 200 estimators, max depth 20
- **Output**: Easy / Medium / Hard

### Regression Model
- **Algorithm**: Gradient Boosting Regressor
- **Parameters**: 200 estimators, max depth 5
- **Output**: Numerical difficulty score

## 📈 Evaluation Metrics

### Classification:
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix

### Regression:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Install missing packages using pip

### Issue: File not found error
**Solution**: Make sure your dataset is named `programming_problems.csv` and is in the same directory

### Issue: Models not loading in app.py
**Solution**: Run `train_model.py` first to generate the model files

### Issue: Low accuracy
**Solution**: 
- Check dataset quality and balance
- Increase training data size
- Tune model hyperparameters
- Add more relevant features

## 📝 Example Usage

```python
# Example problem input:
Title: "Two Sum"
Description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
Input: "First line contains n and target. Second line contains n integers."
Output: "Two space-separated integers representing the indices."

# Expected prediction:
Class: Easy
Score: 800
```

## 🎨 Web Interface Features

- Clean, modern design
- Responsive layout
- Real-time predictions
- Confidence visualization
- Color-coded difficulty levels:
  - 🟢 Easy (Green)
  - 🟡 Medium (Yellow)
  - 🔴 Hard (Red)

## 🔮 Future Enhancements

- Add more advanced NLP features (word embeddings)
- Implement deep learning models (BERT, Transformers)
- Add problem category prediction
- Include time/space complexity prediction
- Support multiple programming languages
- Add batch prediction capability

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Your Name

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- Flask for web framework
- Programming competition platforms for inspiration

---

## 🚨 Important Notes

1. **Dataset Privacy**: Don't share your dataset publicly if it contains copyrighted problems
2. **Model Accuracy**: Results depend on dataset quality and size
3. **Performance**: Initial prediction may take a few seconds
4. **Browser Compatibility**: Best viewed in Chrome, Firefox, or Edge

---

**Need Help?** 
- Check dataset format matches requirements
- Ensure all packages are installed
- Run training script before starting the app
- Check console output for error messages