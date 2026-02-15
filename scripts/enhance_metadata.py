import re

def extract_role_type(profile: str, description: str = "") -> str:
    """Extract role category to prevent semantic drift"""
    profile_lower = profile.lower()
    
    if any(k in profile_lower for k in ['backend', 'back-end', 'server-side']):
        return "backend development"
    if any(k in profile_lower for k in ['frontend', 'front-end', 'ui', 'ux', 'user interface']):
        return "frontend development"
    if any(k in profile_lower for k in ['full stack', 'fullstack']):
        return "full stack development"
    if any(k in profile_lower for k in ['machine learning', 'ml', 'ai', 'deep learning', 'nlp', 'computer vision']):
        return "machine learning / ai"
    if any(k in profile_lower for k in ['data scientist', 'data science', 'data analyst']):
        return "data science / analytics"
    if any(k in profile_lower for k in ['devops', 'sre', 'site reliability']):
        return "devops / infrastructure"
    if any(k in profile_lower for k in ['mobile', 'android', 'ios', 'flutter', 'react native']):
        return "mobile development"
    if any(k in profile_lower for k in ['marketing', 'digital marketing', 'seo', 'social media']):
        return "marketing"
    if any(k in profile_lower for k in ['design', 'graphic', 'ui/ux', 'figma', 'adobe']):
        return "design"
    
    return profile_lower[:50]

def extract_seniority(description: str, profile: str) -> str:
    """Detect experience level"""
    text = (profile + " " + description).lower()
    
    if any(k in text for k in ['senior', 'lead', 'architect', 'principal', '5+ years', 'experienced']):
        return "senior / experienced"
    if any(k in text for k in ['junior', 'entry level', 'entry-level', 'fresher', '0-2 years', 'student', 'internship']):
        return "entry-level / student"
    if any(k in text for k in ['mid', 'mid-level', '2-5 years']):
        return "mid-level"
    
    return "entry-level / student"

def create_embedding_text(row):
    """Create role-context aware embedding text"""
    role_type = extract_role_type(row['profile'], '')
    seniority = extract_seniority('', row['profile'])
    
    skills_text = ", ".join(eval(row['skills_clean'])) if isinstance(row['skills_clean'], str) else "general skills"
    
    return f"""ROLE TYPE: {role_type}
SENIORITY LEVEL: {seniority}
REQUIRED SKILLS: {skills_text}
JOB TITLE: {row['profile']}
COMPANY: {row['company']}
LOCATION: {row['location_normalized']}
DURATION: {row['duration_months']} months
KEYWORDS: internship entry-level student training"""
