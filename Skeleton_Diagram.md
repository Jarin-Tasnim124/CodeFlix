# AI Finder — Skeleton Diagram

This document describes the system architecture of the AI Finder feature.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                   show_enhanced_ai_finder_page()                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CORE CHAT PIPELINE                          │
│                   run_ai_finder_chat_turn()                     │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │
               ▼                                ▼ (if Gemini fails)
┌──────────────────────────┐      ┌─────────────────────────────┐
│     GEMINI AI ENGINE     │      │      SMART FALLBACK         │
│  Generates candidate     │      │  recommender.py             │
│  titles + explanation    │      │  tfidf_recommend()          │
└──────────────┬───────────┘      └──────────────┬──────────────┘
               │                                 │
               └─────────────────┬───────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RECOMMENDATION ENGINE                         │
│   build_explainable_recommendations()                           │
│   rank_recommendations_with_feedback()                          │
│   build_recommendation_reason()                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│   movies.db · recommendation_feedback table · OMDb enrichment   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Modules

These five modules plug into the core pipeline at multiple points:

| Module | What it does | Key functions |
|---|---|---|
| **Explainable recommendations** | Shows a *Why this movie?* reason under each card | `build_recommendation_reason()` ·                      |
| **Like / dislike feedback** | Saves votes to SQLite and biases future rankings | `build_feedback_profile()` ·                      |
| **Quick actions** | One-click prompt templates (mood, similar, anime) | `build_quick_action_prompt()` ·               |
| **Smart fallback** | Replaces Gemini failures with TF-IDF results | `should_use_smart_fallback()                |
| **Recommendation filters** | Year range, IMDb rating, content type (Movie/Anime) | `filter_recommendations_for_ai_finder()` ·               |

---

## End-to-End Flow

```
User input / quick action
        │
        ▼
Filters → prompt instructions + ranking constraints
        │
        ▼
Gemini AI  ──(fail)──▶  TF-IDF fallback (recommender.py)
        │
        ▼
Filter candidates → enrich with "Why this movie?" reasons
        │
        ▼
Rerank with saved like/dislike feedback
        │
        ▼
Render cards  (explanation + feedback buttons + metadata)
```

---

## Primary Files

| File | Responsibility |
|---|---|
| `app.py` | UI, chat flow, feedback persistence, fallback logic, filter logic |
| `recommender.py` | Recommendation reasoning, TF-IDF search, personalization reranking |
| `movies.db` | Stores user feedback in the `recommendation_feedback` table |

---

