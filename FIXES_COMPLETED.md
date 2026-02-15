# Fixes Completed — v2.1.0

## Summary

Fixed **12 issues** across 3 files that caused irrelevant internship recommendations.
Final test accuracy: **100%** (up from ~50% pre-fix).

---

## Critical Fixes (Root Causes)

### 1. Embedding Mismatch → `hybrid_search.py`
**Before:** Query encoded as `"Skill Level: beginner... Seeking: entry-level internship"`.
Document encoded as `"Role: Backend... Skills: Python, Django"`.
→ FAISS compared vectors from different semantic subspaces.

**Fix:** Query now mirrors document format: `"Role: internship\nSkills: {skills}\nLocation: {city}\nEducation: {edu}"`.

### 2. Stale Freshness Scores → `hybrid_search.py`
**Before:** All 8,483 records had `freshness_score = 0.3` (computed once, never updated).
Freshness boost was identical for every record: `0.7×` — pure dead weight.

**Fix:** Removed freshness from scoring formula. All records are from a static 2025 dataset.

### 3. Education Filter Too Strict → `utils.py`
**Before:** Exact string match (`B.Tech != B.E.`). Valid candidates excluded.

**Fix:** Hierarchical check: `B.Tech(4) ≥ Diploma(3) ≥ Any(1)`.

### 4. Stipend Filter Incorrect → `hybrid_search.py`, `lightweight_search.py`
**Before:** Checked `stipend_min < user_min`, filtering out high-paying internships with a low base.

**Fix:** Check `stipend_max < user_min` — if the max offered is below threshold, then reject.

---

## Significant Fixes

### 5. Broken RRF Fusion → `hybrid_search.py`
**Before:** Mixed raw similarity scores with rank-based scores, distorting the fusion.

**Fix:** Pure RRF: `score = 0.6 × 1/(k+rank+1)` (semantic) + `0.4 × 1/(k+rank+1)` (lexical).

### 6. Score Clamping → `hybrid_search.py`
**Before:** `min(1.0, score * 2.0)` clamped all top candidates to 1.0, collapsing ranking.

**Fix:** Normalize by max fused score in the batch.

### 7. Unknown City Distance → `utils.py`
**Before:** Unknown cities got `9999 km` → always filtered out.

**Fix:** Default `300 km` + suburban city mappings (Faridabad→Delhi, Thane→Mumbai, etc.).

### 8. SQL Pre-Sort on Stale Data → `lightweight_search.py`
**Before:** `ORDER BY freshness_score DESC` — all values are `0.3`, wasting sort time.

**Fix:** Removed ORDER BY, increased scan pool to 2000 rows.

---

## Minor Fixes

### 9. Synonym Expansion Truncated → `lightweight_search.py`
**Before:** `break` after first synonym group match. `"python"` only matched Python group, not Data Science.

**Fix:** Removed `break` — checks all synonym groups.

### 10. Silent FTS5 Failures → `hybrid_search.py`
**Before:** Bare `except:` swallowed all errors silently.

**Fix:** `except Exception as e: logging.warning(f"FTS5 search failed: {e}")`.

### 11. No Skill Overlap Scoring → `hybrid_search.py`
**Before:** Semantic and lexical scores only — no direct check for matching skills.

**Fix:** Skill overlap bonus: `1.0 + (overlap / user_skills) × 0.4`.

### 12. Dead Market Rate Code (New) → `hybrid_search.py`
**Before:** `_get_market_rate()` used hardcoded role categories that never matched messy `role_type` values.

**Fix:** Removed market rate calibration entirely.

---

## Test Results (Post-Fix)

```
============================================================
TEST SUMMARY
============================================================
[OK] Lightweight Mode: PASS
[OK] Edge Cases: PASS
[OK] Accuracy: PASS (100.0%)
[OK] Performance: PASS (36ms avg)
[OK] Health Check: PASS

Total: 5/5 tests passed
```

## Files Modified

| File | Changes |
|------|---------|
| `api/utils.py` | Education hierarchy, city distance (300km default + suburban mappings), date parser |
| `api/hybrid_search.py` | Embedding alignment, RRF fusion, education, stipend, scoring, skill overlap, logging |
| `api/lightweight_search.py` | Education post-filter, stipend, SQL scan pool, synonym expansion |
