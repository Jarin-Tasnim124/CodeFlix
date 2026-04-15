"""
Patch script: adds comment box (add/edit/delete) to AI Finder movie cards.
Run once from the CodeFlix directory: py -3 patch_comments.py
"""
import re
import sys

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

APP = "app.py"

with open(APP, "r", encoding="utf-8") as f:
    src = f.read()

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1: Add movie_comments table after the recommendation_feedback indexes
# ──────────────────────────────────────────────────────────────────────────────
OLD1 = (
    "                'CREATE INDEX IF NOT EXISTS idx_recommendation_feedback_created "
    "ON recommendation_feedback(created_at DESC)'\n"
    "            )\n"
    "            \n"
    "            # Check if database needs initial setup marker"
)

NEW1 = (
    "                'CREATE INDEX IF NOT EXISTS idx_recommendation_feedback_created "
    "ON recommendation_feedback(created_at DESC)'\n"
    "            )\n"
    "\n"
    "            # Movie Comments table (AI Finder)\n"
    "            c.execute(\n"
    "                '''\n"
    "                CREATE TABLE IF NOT EXISTS movie_comments (\n"
    "                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    "                    movie_title TEXT NOT NULL,\n"
    "                    comment_text TEXT NOT NULL,\n"
    "                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
    "                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
    "                )\n"
    "                '''\n"
    "            )\n"
    "            c.execute(\n"
    "                'CREATE INDEX IF NOT EXISTS idx_movie_comments_title ON movie_comments(movie_title)'\n"
    "            )\n"
    "            c.execute(\n"
    "                'CREATE INDEX IF NOT EXISTS idx_movie_comments_created ON movie_comments(created_at DESC)'\n"
    "            )\n"
    "\n"
    "            # Check if database needs initial setup marker"
)

if OLD1 not in src:
    print("PATCH 1 target not found - checking actual text ...")
    idx = src.find("idx_recommendation_feedback_created")
    print(repr(src[idx-5:idx+250]))
    sys.exit(1)

src = src.replace(OLD1, NEW1, 1)
print("PATCH 1 applied (movie_comments table)")

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2: Wire comment section + inject CRUD helpers & render function
# ──────────────────────────────────────────────────────────────────────────────
OLD2 = (
    '                    add_movie(movie["title"], movie.get("genre", "Unknown"), year_value, False)\n'
    '                    st.success(f"Added {movie[\'title\']}!")\n'
    "\n"
    "# -----------------------------\n"
    "# Enhanced Database Functions\n"
    "# -----------------------------"
)

CRUD_AND_RENDER = r'''
# ------------------------------
# Movie Comments - CRUD Helpers
# ------------------------------
def get_movie_comments(movie_title):
    """Return all comments for a movie, newest first."""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT id, comment_text, created_at, updated_at FROM movie_comments "
                "WHERE movie_title = ? ORDER BY created_at DESC",
                (movie_title,)
            )
            rows = c.fetchall()
            return [
                {"id": r[0], "text": r[1], "created_at": r[2], "updated_at": r[3]}
                for r in rows
            ]
    except sqlite3.Error as e:
        logging.error(f"get_movie_comments error: {e}")
        return []


def add_movie_comment(movie_title, comment_text):
    """Insert a new comment. Returns True on success."""
    if not comment_text.strip():
        return False
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO movie_comments (movie_title, comment_text) VALUES (?, ?)",
                (movie_title, comment_text.strip())
            )
            conn.commit()
            return True
    except sqlite3.Error as e:
        logging.error(f"add_movie_comment error: {e}")
        return False


def update_movie_comment(comment_id, new_text):
    """Update an existing comment. Returns True on success."""
    if not new_text.strip():
        return False
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE movie_comments SET comment_text = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_text.strip(), comment_id)
            )
            conn.commit()
            return c.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"update_movie_comment error: {e}")
        return False


def delete_movie_comment(comment_id):
    """Delete a comment by id. Returns True on success."""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM movie_comments WHERE id = ?", (comment_id,))
            conn.commit()
            return c.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"delete_movie_comment error: {e}")
        return False


def render_movie_comments_section(movie_title, key_prefix):
    """
    Render a collapsible comment box below a movie card in AI Finder.
    - Shows existing comments with Edit / Delete buttons.
    - Provides a textarea + Add Comment button at the bottom.
    """
    safe_key = re.sub(r'[^a-zA-Z0-9]', '_', movie_title)[:40]
    widget_key = f"{key_prefix}_{safe_key}"

    with st.expander(f"Comments for **{movie_title}**", expanded=False):
        comments = get_movie_comments(movie_title)

        # flash messages
        flash_key = f"cmnt_flash_{widget_key}"
        if st.session_state.get(flash_key):
            st.success(st.session_state.pop(flash_key))

        # existing comments
        if comments:
            for cmt in comments:
                cmt_id   = cmt["id"]
                cmt_text = cmt["text"]
                ts       = cmt["created_at"][:16] if cmt["created_at"] else ""
                edited   = " (edited)" if cmt["updated_at"] != cmt["created_at"] else ""

                edit_flag_key = f"cmnt_edit_{widget_key}_{cmt_id}"
                edit_text_key = f"cmnt_etext_{widget_key}_{cmt_id}"

                # styled comment card
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.05);border-left:3px solid #FFD93D;
                                padding:0.5rem 0.75rem;border-radius:6px;margin-bottom:0.5rem;">
                        <span style="color:#eee;font-size:0.95rem;">{html.escape(cmt_text)}</span><br>
                        <span style="color:#888;font-size:0.72rem;">{ts}{edited}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Edit / Delete row
                btn_cols = st.columns([1, 1, 4])
                with btn_cols[0]:
                    if st.button("Edit", key=f"btn_edit_{widget_key}_{cmt_id}", use_container_width=True):
                        st.session_state[edit_flag_key] = True
                        st.session_state[edit_text_key] = cmt_text
                        st.rerun()
                with btn_cols[1]:
                    if st.button("Delete", key=f"btn_del_{widget_key}_{cmt_id}", use_container_width=True):
                        if delete_movie_comment(cmt_id):
                            st.session_state[flash_key] = "Comment deleted."
                        st.rerun()

                # Inline edit form
                if st.session_state.get(edit_flag_key):
                    new_text = st.text_area(
                        "Edit comment",
                        value=st.session_state.get(edit_text_key, cmt_text),
                        key=f"ta_edit_{widget_key}_{cmt_id}",
                        height=90,
                    )
                    save_cancel = st.columns(2)
                    with save_cancel[0]:
                        if st.button("Save", key=f"btn_save_{widget_key}_{cmt_id}",
                                     use_container_width=True, type="primary"):
                            if update_movie_comment(cmt_id, new_text):
                                st.session_state[flash_key] = "Comment updated!"
                                st.session_state.pop(edit_flag_key, None)
                                st.session_state.pop(edit_text_key, None)
                            st.rerun()
                    with save_cancel[1]:
                        if st.button("Cancel", key=f"btn_cancel_{widget_key}_{cmt_id}",
                                     use_container_width=True):
                            st.session_state.pop(edit_flag_key, None)
                            st.session_state.pop(edit_text_key, None)
                            st.rerun()
        else:
            st.caption("No comments yet. Be the first to comment!")

        # Add new comment
        st.markdown("---")
        new_comment = st.text_area(
            "Write a comment...",
            placeholder=f"Share your thoughts about {movie_title}...",
            key=f"ta_new_{widget_key}",
            height=80,
        )
        if st.button("Add Comment", key=f"btn_add_{widget_key}",
                     use_container_width=True, type="primary"):
            if add_movie_comment(movie_title, new_comment):
                st.session_state[flash_key] = "Comment added!"
            else:
                st.warning("Comment cannot be empty.")
            st.rerun()

'''

NEW2 = (
    '                    add_movie(movie["title"], movie.get("genre", "Unknown"), year_value, False)\n'
    '                    st.success(f"Added {movie[\'title\']}!")\n'
    "\n"
    "        # Comment section below every recommended movie\n"
    "        render_movie_comments_section(\n"
    "            movie_title=movie.get(\"title\", f\"movie_{index}\"),\n"
    "            key_prefix=f\"{key_prefix}_cmt_{index}\",\n"
    "        )\n"
    + CRUD_AND_RENDER +
    "\n# -----------------------------\n"
    "# Enhanced Database Functions\n"
    "# -----------------------------"
)

if OLD2 not in src:
    print("PATCH 2 target not found - showing context ...")
    idx = src.find('add_movie(movie["title"]')
    print(repr(src[idx-5:idx+400]))
    sys.exit(1)

src = src.replace(OLD2, NEW2, 1)
print("PATCH 2 applied (CRUD helpers + render_movie_comments_section + wire-up)")

# ──────────────────────────────────────────────────────────────────────────────
# Write result
# ──────────────────────────────────────────────────────────────────────────────
with open(APP, "w", encoding="utf-8") as f:
    f.write(src)

print("All patches applied. Restart Streamlit to see changes.")
