# BGE-M3 Semantic Search Testing

## Quick Start

### Option 1: Detailed Output (Console)
```bash
python test_bge_m3.py
```
Shows detailed results for 10 student profiles in console with formatting.

### Option 2: Simple Output (File)
```bash
python test_bge_m3_simple.py
```
Saves results to `bge_m3_results.txt` for easy review.

## Requirements

```bash
pip install sentence-transformers faiss-cpu torch
```

## Test Profiles (10 Students)

1. **Rahul** - Backend Developer (Python, Django, REST API)
2. **Priya** - Frontend Developer (React, JavaScript, HTML)
3. **Amit** - Data Scientist (Python, ML, Pandas)
4. **Sneha** - Digital Marketer (Social Media, Content, SEO)
5. **Karthik** - Full Stack Developer (MERN Stack)
6. **Ananya** - UI/UX Designer (Figma, UI Design)
7. **Rohan** - Mobile Developer (React Native, Android)
8. **Divya** - Content Creator (Content Writing)
9. **Arjun** - DevOps Engineer (Docker, Kubernetes, AWS)
10. **Meera** - Business Analyst (Excel, SQL, Tableau)

## What Gets Tested

- ✅ Semantic understanding (e.g., "React" → "Frontend Development")
- ✅ Skill variations (e.g., "ML" → "Machine Learning")
- ✅ Context awareness (e.g., "Python for web" vs "Python for data")
- ✅ Hybrid scoring (70% semantic + 30% keyword)
- ✅ Location filtering (distance-based)
- ✅ Education matching
- ✅ Stipend filtering

## Expected Performance

- **Accuracy**: 85-95% (vs 62.5% in lightweight mode)
- **Latency**: 50-100ms per query
- **Memory**: ~2.3 GB (BGE-M3 model)
- **Startup**: ~5 seconds (model loading)

## Output Format

Each student profile shows:
- Top 5-10 internship recommendations
- Match score (0-100)
- Distance from student location
- Stipend range
- Required skills
- Company name

## Comparison: Lightweight vs BGE-M3

| Feature | Lightweight | BGE-M3 |
|---------|-------------|--------|
| Accuracy | 62.5% | 85-95% |
| Memory | 512 MB | 2.3 GB |
| Latency | 16ms | 50-100ms |
| Semantic | ❌ No | ✅ Yes |
| Context | ❌ No | ✅ Yes |

## Troubleshooting

**Error: "FAISS not installed"**
```bash
pip install faiss-cpu
```

**Error: "sentence-transformers not found"**
```bash
pip install sentence-transformers
```

**Out of Memory**
- Close other applications
- Use lightweight mode instead
- Reduce batch size in code

## Files

- `test_bge_m3.py` - Detailed console output
- `test_bge_m3_simple.py` - Simple file output
- `bge_m3_results.txt` - Generated results file
