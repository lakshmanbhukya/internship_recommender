import pandas as pd
import sys
sys.path.append('.')
from scripts.enhance_metadata import extract_role_type, extract_seniority, create_embedding_text

print("Loading dataset...")
df = pd.read_csv('data/processed/internships_cleaned.csv')

print(f"Enhancing {len(df)} internships with role_type and seniority...")
df['role_type'] = df.apply(lambda r: extract_role_type(r['profile'], ''), axis=1)
df['seniority'] = df.apply(lambda r: extract_seniority('', r['profile']), axis=1)

print("Regenerating embedding_text with new structure...")
df['embedding_text'] = df.apply(create_embedding_text, axis=1)

output_path = 'data/processed/internships_enhanced.csv'
df.to_csv(output_path, index=False)
print(f"[SUCCESS] Enhanced dataset saved to {output_path}")
print("\nSample enhanced embedding:")
print(df['embedding_text'].iloc[0])
