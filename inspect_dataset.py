import json
import pandas as pd

# ============================================
# INSPECT JSONL DATASET
# ============================================
print("="*60)
print("JSONL Dataset Inspector")
print("="*60)

# Read JSONL file
print("\n📂 Loading dataset...")
data_list = []

try:
    with open('programming_problems.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line {i+1}: {e}")
                
    print(f"✅ Successfully loaded {len(data_list)} records")
    
except FileNotFoundError:
    print("❌ Error: 'programming_problems.jsonl' not found!")
    print("Please place your JSONL file in the same directory as this script.")
    exit()

# Convert to DataFrame
df = pd.DataFrame(data_list)

# ============================================
# BASIC INFORMATION
# ============================================
print("\n" + "="*60)
print("📊 BASIC INFORMATION")
print("="*60)

print(f"\nTotal records: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# ============================================
# COLUMN DETAILS
# ============================================
print("\n" + "="*60)
print("📋 COLUMN DETAILS")
print("="*60)

for col in df.columns:
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Non-null: {df[col].notna().sum()}/{len(df)}")
    print(f"  Null: {df[col].isna().sum()}")
    
    if df[col].dtype == 'object':
        print(f"  Unique values: {df[col].nunique()}")
        if df[col].nunique() < 20:
            print(f"  Values: {df[col].unique().tolist()}")

# ============================================
# CHECK REQUIRED COLUMNS
# ============================================
print("\n" + "="*60)
print("✅ REQUIRED COLUMNS CHECK")
print("="*60)

required_columns = [
    'title',
    'description', 
    'input_description',
    'output_description',
    'problem_class',
    'problem_score'
]

print("\nChecking for required columns...")
missing_columns = []

for col in required_columns:
    if col in df.columns:
        print(f"  ✅ {col} - Found")
    else:
        print(f"  ❌ {col} - MISSING")
        missing_columns.append(col)

if missing_columns:
    print(f"\n⚠️  WARNING: Missing columns: {missing_columns}")
    print("Please check your dataset or let me know the actual column names.")
else:
    print("\n✅ All required columns found!")

# ============================================
# CLASS DISTRIBUTION
# ============================================
if 'problem_class' in df.columns:
    print("\n" + "="*60)
    print("📊 CLASS DISTRIBUTION")
    print("="*60)
    print("\n", df['problem_class'].value_counts())
    print("\nPercentages:")
    print(df['problem_class'].value_counts(normalize=True) * 100)

# ============================================
# SCORE STATISTICS
# ============================================
if 'problem_score' in df.columns:
    print("\n" + "="*60)
    print("📊 SCORE STATISTICS")
    print("="*60)
    print("\n", df['problem_score'].describe())

# ============================================
# SAMPLE DATA
# ============================================
print("\n" + "="*60)
print("📄 SAMPLE DATA (First Record)")
print("="*60)

if len(data_list) > 0:
    sample = data_list[0]
    print("\n", json.dumps(sample, indent=2, ensure_ascii=False)[:1000])
    if len(json.dumps(sample, indent=2)) > 1000:
        print("\n... (truncated)")

# ============================================
# TEXT LENGTH ANALYSIS
# ============================================
print("\n" + "="*60)
print("📏 TEXT LENGTH ANALYSIS")
print("="*60)

text_columns = ['title', 'description', 'input_description', 'output_description']
for col in text_columns:
    if col in df.columns:
        df[f'{col}_length'] = df[col].fillna('').astype(str).str.len()
        print(f"\n{col}:")
        print(f"  Avg length: {df[f'{col}_length'].mean():.0f} chars")
        print(f"  Max length: {df[f'{col}_length'].max():.0f} chars")
        print(f"  Min length: {df[f'{col}_length'].min():.0f} chars")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("📝 SUMMARY")
print("="*60)

print(f"\n✅ Dataset is ready for training!")
print(f"   - Total problems: {len(df)}")
if 'problem_class' in df.columns:
    print(f"   - Classes: {df['problem_class'].nunique()}")
if 'problem_score' in df.columns:
    print(f"   - Score range: {df['problem_score'].min()} - {df['problem_score'].max()}")

print("\n💡 Next steps:")
print("   1. If all columns are present, run: python train_model.py")
print("   2. If columns are missing, share the output with me")
print("   3. After training, run: python app.py")

print("\n" + "="*60)