import json
import pandas as pd

# Load your data
data = []
with open('programming_problems.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Check class labels
print("="*50)
print("YOUR DATASET CLASS LABELS:")
print("="*50)
print("\nUnique classes:", df['problem_class'].unique())
print("\nClass counts:")
print(df['problem_class'].value_counts())
print("\nSample of first 5 class values:")
print(df['problem_class'].head(10).tolist())