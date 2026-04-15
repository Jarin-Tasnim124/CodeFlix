# AI Finder Feature Implementation

This document summarizes the AI Finder features implemented from [AI_COMPONENT_IMPLEMENTATION_PLAN_5_DAYS.md](AI_COMPONENT_IMPLEMENTATION_PLAN_5_DAYS.md), with code references for the main integration points.

## Scope

Implemented feature areas:

1. Explainable recommendations
2. Like/Dislike feedback loop with edit/delete management
3. Prompt templates (quick actions)
4. Smart fallback recommender
5. Recommendation filters

Core AI Finder entry point:

- `app.py:2504` - `show_enhanced_ai_finder_page()`

Recommendation engine helpers:

- `app.py:1099` - `build_explainable_recommendations(...)`
- `recommender.py:295` - `build_recommendation_reason(...)`
- `recommender.py:352` - `add_recommendation_reasons(...)`

## 1. Explainable Recommendations

### What it does

- Shows a short `Why this movie?` explanation under each recommendation
- Generates reasons such as mood match, genre match, actor match, and theme similarity

### Main code references

- `app.py:1099` - `build_explainable_recommendations(...)`
- `app.py:1580` - `render_explainable_recommendations(...)`
- `recommender.py:247` - `_extract_query_signals(...)`
- `recommender.py:295` - `build_recommendation_reason(...)`
- `recommender.py:352` - `add_recommendation_reasons(...)`

### Behavior

- Recommendation reasons are built from the user prompt plus local or OMDb-enriched movie details
- The explanation renderer is reused across:
  - AI chat recommendations
  - smart fallback recommendations
  - no-results recommendation cards
- Example reason styles include:
  - `Mood match`
  - `Genre match`
  - `Actor match`
  - `Theme similarity`

## 2. Like/Dislike Feedback Loop

### What it does

- Adds `like` and `dislike` controls to recommendation cards in AI Finder
- Saves feedback in SQLite
- Lets users edit or delete saved recommendation feedback from the AI Finder page
- Uses saved feedback to bias future recommendations

### Main code references

- `app.py:1312` - reads recent feedback rows
- `app.py:1333` - builds feedback context and latest vote map
- `app.py:1347` - saves feedback for a recommendation
- `app.py:1375` - updates a saved feedback row
- `app.py:1405` - deletes a saved feedback row
- `app.py:1438` - builds personalization summary text for the UI
- `app.py:1649` - renders recommendation cards with feedback buttons
- `app.py:1739` - renders feedback edit/delete management UI
- `app.py:1801` - creates `recommendation_feedback` table
- `recommender.py:116` - `build_feedback_profile(...)`
- `recommender.py:186` - `rank_recommendations_with_feedback(...)`

### Behavior

- A clicked vote is stored with movie title, genre, year, vote type, source query, and timestamp
- Users can update a saved vote, title, genre, year, or source prompt from the `Manage Saved Recommendation Feedback` panel
- Users can delete a saved feedback row when they want to remove it from personalization memory
- Saved likes/dislikes influence later ranking by title and genre
- Recommendation cards show the current vote state when available

## 3. Prompt Templates (Quick Actions)

### What it does

- Adds one-click quick prompts inside the AI Finder chat
- Supports:
  - Recommend by mood
  - Similar to this movie
  - Anime-only recommendations

### Main code references

- `app.py:1436` - `build_quick_action_prompt(...)`
- `app.py:1527` - `run_ai_finder_chat_turn(...)`
- `app.py:2368` - quick action UI block inside `show_enhanced_ai_finder_page()`

### Behavior

- Quick actions submit through the same pipeline as normal typed chat input
- `Similar to This Movie` and `Anime-Only` reuse the left search input when available
- Quick action prompts still benefit from feedback personalization and filter logic

## 4. Smart Fallback Recommender

### What it does

- Detects Gemini failure or unusable AI responses
- Switches to TF-IDF recommendations from `recommender.py`
- Replaces generic failure text with a meaningful fallback message

### Main code references

- `app.py:1451` - `should_use_smart_fallback(...)`
- `app.py:1474` - `build_smart_fallback_recommendations(...)`
- `app.py:1500` - `build_smart_fallback_message(...)`
- `app.py:1527` - fallback is triggered in `run_ai_finder_chat_turn(...)`
- `recommender.py:366` - `tfidf_recommend(...)`

### Behavior

- Fallback is used when Gemini:
  - is unavailable
  - returns an error
  - returns no candidate titles
  - returns an empty reply
- Fallback recommendations still pass through explainable recommendation rendering
- Saved user feedback still affects fallback ranking

## 5. Recommendation Filters

### What it does

- Adds optional AI Finder filters for:
  - Year range
  - Minimum IMDb rating
  - Content type (`All`, `Movie`, `Anime`)

### Main code references

- `app.py:1167` - `get_ai_finder_filter_defaults()`
- `app.py:1184` - `build_ai_filter_summary(...)`
- `app.py:1204` - `build_ai_filter_prompt_context(...)`
- `app.py:1226` - `recommendation_matches_basic_filters(...)`
- `app.py:1261` - `filter_recommendations_for_ai_finder(...)`
- `app.py:1385` - filtered feedback-aware ranking
- `app.py:1474` - filtered fallback TF-IDF ranking
- `app.py:2268` - filter UI setup inside `show_enhanced_ai_finder_page()`
- `app.py:2687` - filtered no-results recommendation fallback

### Behavior

- Filters are shown in an expandable panel above AI Finder
- Filters affect:
  - Gemini prompt instructions
  - AI recommendation ranking
  - TF-IDF fallback results
  - no-results recommendation cards
- Recommendation cards can show extra metadata such as IMDb rating and content type when available

### Note

- The IMDb filter works best when OMDb data is reachable
- In restricted/offline conditions, a strict IMDb threshold may reduce or eliminate matches instead of guessing unsupported ratings

## End-to-End Flow

1. User enters a prompt or clicks a quick action in AI Finder
2. Active filters are converted into prompt instructions and ranking constraints
3. Gemini returns explanation text and candidate titles
4. If Gemini fails, the app switches to TF-IDF fallback
5. Candidate recommendations are filtered, enriched with `Why this movie?` reasons, and reranked with saved feedback
6. Cards are rendered with explanations, feedback buttons, and optional metadata

## Primary Files

- `app.py` - UI, chat flow, feedback persistence, fallback logic, filter logic
- `recommender.py` - recommendation reasoning, TF-IDF search, personalization reranking
- `movies.db` - stores feedback in `recommendation_feedback`
