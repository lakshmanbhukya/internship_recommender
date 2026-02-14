import pandas as pd
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *

def normalize_city(city_name):
    if pd.isna(city_name):
        return "Unknown"
    city = city_name.strip()
    if city in CITY_MAPPINGS:
        return CITY_MAPPINGS[city]
    for key, value in CITY_MAPPINGS.items():
        if key.lower() in city.lower():
            return value
    if any(kw in city.lower() for kw in ['work from home', 'remote', 'anywhere']):
        return "Remote"
    return city

def parse_stipend(stipend_str):
    if pd.isna(stipend_str) or 'not' in str(stipend_str).lower():
        return DEFAULT_STIPEND
    numbers = re.findall(r'\d+', str(stipend_str).replace(',', ''))
    if not numbers:
        return DEFAULT_STIPEND
    min_s = int(numbers[0])
    max_s = int(numbers[1]) if len(numbers) > 1 else min_s
    return {"min": min_s, "max": max_s}

def parse_skills(skills_str):
    if pd.isna(skills_str) or skills_str == "":
        return []
    skills = re.split(r'[,;|/]+', str(skills_str))
    return [s.strip() for s in skills if s.strip() and len(s.strip()) > 1]

def normalize_education(edu_str):
    if pd.isna(edu_str):
        return "Any"
    edu = edu_str.strip()
    if edu in EDU_MAPPINGS:
        return EDU_MAPPINGS[edu]
    for key, value in EDU_MAPPINGS.items():
        if key.lower() in edu.lower():
            return value
    return "Any"

def calculate_freshness(date_str):
    try:
        post_date = pd.to_datetime(date_str)
        days_old = (datetime.now() - post_date).days
        return max(0.3, 1.0 - days_old / 30.0)
    except:
        return 1.0

def parse_duration(duration_str):
    if pd.isna(duration_str):
        return DEFAULT_DURATION
    numbers = re.findall(r'\d+', str(duration_str))
    return int(numbers[0]) if numbers else DEFAULT_DURATION

def create_embedding_text(row):
    skills_text = ", ".join(row['skills_clean']) if row['skills_clean'] else "No specific skills"
    return f"""Role: {row['profile']}
Skills: {skills_text}
Company: {row['company']}
Location: {row['location_normalized']}
Duration: {row['duration_months']} months
Education: {row['education_normalized']}
Perks: {row['Perks'] if pd.notna(row['Perks']) else 'Standard benefits'}"""

def preprocess_internships():
    print("🚀 Starting preprocessing...")
    
    csv_file = RAW_DATA / "merged_internships_dataset.csv"
    if not csv_file.exists():
        print("❌ merged_internships_dataset.csv not found in data/raw/")
        print("Run: python scripts/download_dataset.py first")
        return None
    
    print(f"📂 Loading {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"📊 Original: {df.shape}")
    
    df = df.drop_duplicates(subset=['internship_id'])
    print(f"🧹 After dedup: {df.shape}")
    
    print("📍 Normalizing locations...")
    df['location_normalized'] = df['Location'].apply(normalize_city)
    
    print("💰 Parsing stipend...")
    stipend_info = df['Stipend'].apply(parse_stipend)
    df['stipend_min'] = stipend_info.apply(lambda x: x['min'])
    df['stipend_max'] = stipend_info.apply(lambda x: x['max'])
    
    print("🛠️ Parsing skills...")
    df['skills_clean'] = df['Skills'].apply(parse_skills)
    
    print("🎓 Normalizing education...")
    df['education_normalized'] = df['Education'].apply(normalize_education)
    
    print("⏱️ Parsing duration...")
    df['duration_months'] = df['Duration'].apply(parse_duration)
    
    print("🕐 Calculating freshness...")
    df['freshness_score'] = df['Date Time'].apply(calculate_freshness)
    
    print("📝 Creating embedding text...")
    df['embedding_text'] = df.apply(create_embedding_text, axis=1)
    
    final_cols = [
        'internship_id', 'profile', 'company', 'Location', 'location_normalized',
        'stipend_min', 'stipend_max', 'duration_months', 'education_normalized',
        'skills_clean', 'Perks', 'Apply by Date', 'freshness_score', 'embedding_text'
    ]
    df_final = df[final_cols].copy()
    
    print(f"💾 Saving to {PROCESSED_DATA}")
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(PROCESSED_DATA, index=False)
    
    print(f"✅ Complete! Final: {df_final.shape}")
    print(f"\n📊 Summary:")
    print(f"Total: {len(df_final)}")
    print(f"Cities: {df_final['location_normalized'].nunique()}")
    print(f"Top cities: {df_final['location_normalized'].value_counts().head(5).to_dict()}")
    
    return df_final

if __name__ == "__main__":
    preprocess_internships()
