# AI Finder Feature Implementation

This document summarizes the AI Finder features implemented from [AI_COMPONENT_IMPLEMENTATION_PLAN_5_DAYS.md](AI_COMPONENT_IMPLEMENTATION_PLAN_5_DAYS.md), with code references for the main integration points.

## Scope

Implemented feature areas:

1. Explainable recommendations
2. Like/Dislike feedback loop
3. Prompt templates (quick actions)
4. Smart fallback recommender
5. Recommendation filters
6. Chat history edit & delete
7. Movie comment box (Add / Edit / Delete)

Core AI Finder entry point:

- `app.py` - `show_enhanced_ai_finder_page()`

Recommendation engine helpers:

- `app.py` - `build_explainable_recommendations(...)`
- `recommender.py` - `build_recommendation_reason(...)`
- `recommender.py` - `add_recommendation_reasons(...)`

## 1. Explainable Recommendations

### What it does

- Shows a short `Why this movie?` explanation under each recommendation
- Generates reasons such as mood match, genre match, actor match, and theme similarity

### Main code references

- `app.py` - `build_explainable_recommendations(...)`
- `app.py` - `render_explainable_recommendations(...)`
- `recommender.py` - `_extract_query_signals(...)`
- `recommender.py` - `build_recommendation_reason(...)`
- `recommender.py` - `add_recommendation_reasons(...)`

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
- Uses saved feedback to bias future recommendations

### Main code references

- `app.py` - creates `recommendation_feedback` table in `init_db()`
- `app.py` - reads recent feedback rows
- `app.py` - builds feedback context and latest vote map
- `app.py` - saves feedback for a recommendation (`save_recommendation_feedback`)
- `app.py` - builds personalization summary text for the UI
- `app.py` - renders recommendation cards with feedback buttons (`render_explainable_recommendations`)
- `recommender.py` - `build_feedback_profile(...)`
- `recommender.py` - `rank_recommendations_with_feedback(...)`

### Behavior

- A clicked vote is stored with movie title, genre, year, vote type, source query, and timestamp
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

- `app.py` - `build_quick_action_prompt(...)`
- `app.py` - `run_ai_finder_chat_turn(...)`
- `app.py` - quick action UI block inside `show_enhanced_ai_finder_page()`

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

- `app.py` - `should_use_smart_fallback(...)`
- `app.py` - `build_smart_fallback_recommendations(...)`
- `app.py` - `build_smart_fallback_message(...)`
- `app.py` - fallback is triggered in `run_ai_finder_chat_turn(...)`
- `recommender.py` - `tfidf_recommend(...)`

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

- `app.py` - `get_ai_finder_filter_defaults()`
- `app.py` - `build_ai_filter_summary(...)`
- `app.py` - `build_ai_filter_prompt_context(...)`
- `app.py` - `recommendation_matches_basic_filters(...)`
- `app.py` - `filter_recommendations_for_ai_finder(...)`
- `app.py` - filtered feedback-aware ranking
- `app.py` - filtered fallback TF-IDF ranking
- `app.py` - filter UI setup inside `show_enhanced_ai_finder_page()`

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

## 6. Chat History Edit & Delete

### What it does

- Allows users to **edit** a previously sent prompt in the AI Finder chat
- Regenerates the AI response when the edited prompt is saved
- Allows users to **delete** any chat turn (prompt + response pair) from history

### Main code references

- `app.py` - `sync_ai_finder_chat_history(chat)` — syncs in-memory history with session state
- `app.py` - `get_ai_finder_history_turns()` — returns all chat turns for rendering
- `app.py` - `clear_ai_finder_edit_state()` — clears the active edit selection
- `app.py` - `delete_ai_finder_history_turn(chat, user_index)` — removes a turn from history
- `app.py` - `update_ai_finder_history_turn(...)` — re-runs Gemini with the edited prompt and stores the result
- `app.py` - Edit/Delete UI block inside the chat history loop in `show_enhanced_ai_finder_page()`

### Behavior

- Each chat bubble in the history shows **Edit** and **Delete** buttons below the user message
- Clicking **Edit** opens an inline text area pre-filled with the original prompt
- **Save Changes** re-calls Gemini and replaces the old assistant response in history
- **Cancel** closes the edit form without changes
- Clicking **Delete** removes the entire turn (user message + assistant response) and shows a flash confirmation
- Flash messages are scoped per-turn to avoid cross-turn interference

## 7. Movie Comment Box

### What it does

- Adds a collapsible **comment section** below each recommended movie card in AI Finder chat results
- Supports full **Add / Edit / Delete** operations on comments
- Comments are stored persistently in SQLite and scoped per movie title

### Main code references

- `app.py` - `movie_comments` table created in `init_db()`
- `app.py` - `get_movie_comments(movie_title)` — fetches all comments for a movie, newest first
- `app.py` - `add_movie_comment(movie_title, comment_text)` — inserts a new comment
- `app.py` - `update_movie_comment(comment_id, new_text)` — updates comment text and `updated_at` timestamp
- `app.py` - `delete_movie_comment(comment_id)` — removes a comment by ID
- `app.py` - `render_movie_comments_section(movie_title, key_prefix)` — renders the full comment UI
- `app.py` - called from `render_explainable_recommendations(...)` after the 👍/👎 buttons for every movie

### Database schema

```sql
CREATE TABLE movie_comments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_title  TEXT NOT NULL,
    comment_text TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Behavior

- The comment section is hidden by default inside a **collapsible expander** to keep the UI clean
- When expanded, existing comments are shown as styled cards with a yellow left border
- Each comment card displays:
  - The comment text
  - The timestamp (`created_at`)
  - An `(edited)` label if the comment has been modified
  - **✏️ Edit** and **🗑️ Delete** buttons — equal-width, flush side-by-side (no empty gaps)
- Clicking **✏️ Edit** opens an inline text area below the card with **💾 Save** and **Cancel** buttons
- Clicking **🗑️ Delete** immediately removes the comment and shows a flash confirmation
- The **➕ Add Comment** input area is always visible at the bottom of the expander
- Flash messages (added / updated / deleted) are scoped per movie to avoid cross-card interference
- Comments are shared across all chat sessions since they are stored in the database (not session state)

## End-to-End Flow

1. User enters a prompt or clicks a quick action in AI Finder
2. Active filters are converted into prompt instructions and ranking constraints
3. Gemini returns explanation text and candidate titles
4. If Gemini fails, the app switches to TF-IDF fallback
5. Candidate recommendations are filtered, enriched with `Why this movie?` reasons, and reranked with saved feedback
6. Cards are rendered with explanations, feedback buttons (👍/👎), and optional metadata
7. Below each card, a **Comments** expander allows the user to add, edit, or delete personal notes about that movie
8. The user can edit or delete any past chat turn from the history without leaving the page

## Primary Files

- `app.py` — UI, chat flow, feedback persistence, fallback logic, filter logic, chat edit/delete, comment box CRUD
- `recommender.py` — recommendation reasoning, TF-IDF search, personalization reranking
- `movies.db` — stores data in:
  - `recommendation_feedback` — like/dislike votes per movie
  - `movie_comments` — user comments per movie (Add / Edit / Delete)

