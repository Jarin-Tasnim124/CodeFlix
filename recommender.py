# recommender.py - AI movie recommendation system
import pandas as pd
import requests
import os
import random
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


GENRE_KEYWORDS = {
    'action': ['action', 'adventure', 'fight', 'superhero', 'explosive'],
    'comedy': ['comedy', 'funny', 'humor', 'laugh', 'feel good'],
    'drama': ['drama', 'emotional', 'serious', 'character'],
    'romance': ['romance', 'romantic', 'love', 'heart'],
    'sci-fi': ['sci-fi', 'sci fi', 'science fiction', 'space', 'future', 'time travel'],
    'thriller': ['thriller', 'tense', 'mystery', 'suspense'],
    'horror': ['horror', 'scary', 'terror', 'haunted', 'monster'],
    'anime': ['anime', 'manga', 'japanese', 'shonen'],
    'animation': ['animation', 'animated', 'family'],
}

THEME_KEYWORDS = {
    'space': ['space', 'galaxy', 'cosmic', 'astronaut', 'wormhole'],
    'mind-bending': ['dream', 'mind-bending', 'mind bending', 'twist', 'psychological'],
    'crime': ['crime', 'mafia', 'gangster', 'heist', 'detective'],
    'survival': ['survival', 'post-apocalyptic', 'post apocalyptic', 'escape'],
    'friendship': ['friendship', 'friends', 'bond', 'team'],
    'revenge': ['revenge', 'vengeance', 'payback'],
    'coming-of-age': ['coming of age', 'growing up', 'teen'],
    'hero-journey': ['hero', 'chosen one', 'quest'],
}

MOOD_PROFILES = {
    'sad': {
        'keywords': ['sad', 'lonely', 'heartbroken', 'melancholy', 'cry'],
        'genres': ['drama', 'romance'],
        'descriptor': 'emotional',
    },
    'happy': {
        'keywords': ['happy', 'feel good', 'uplifting', 'cheerful'],
        'genres': ['comedy', 'animation', 'adventure', 'musical'],
        'descriptor': 'feel-good',
    },
    'tense': {
        'keywords': ['tense', 'intense', 'edge of my seat', 'thrilling'],
        'genres': ['thriller', 'action', 'crime', 'horror'],
        'descriptor': 'high-stakes',
    },
    'thoughtful': {
        'keywords': ['thoughtful', 'deep', 'philosophical', 'smart'],
        'genres': ['sci-fi', 'drama', 'biography'],
        'descriptor': 'thought-provoking',
    },
    'adventurous': {
        'keywords': ['adventurous', 'epic', 'exciting', 'fun ride'],
        'genres': ['action', 'adventure', 'fantasy', 'anime'],
        'descriptor': 'adventurous',
    },
}

ACTOR_PATTERNS = [
    r"(?:with|starring|stars?|actor)\s+([a-z][a-z .'-]+)",
]

REFERENCE_TITLE_PATTERNS = [
    r"(?:like|similar to)\s+([a-z0-9][^,.;!?\n]+)",
]

TOKEN_STOPWORDS = {
    'movie', 'movies', 'film', 'films', 'show', 'shows', 'recommend', 'recommendation',
    'recommendations', 'watch', 'something', 'want', 'need', 'please', 'like', 'similar',
    'more', 'less', 'with', 'about', 'that', 'this', 'from', 'for', 'and', 'the', 'a', 'an',
}


def _normalize_title_key(title):
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _extract_genre_tokens(genre_value):
    return [
        token.strip().lower()
        for token in re.split(r"[,/|]", genre_value or "")
        if token.strip()
    ]


def _normalize_movie_record(movie):
    """Return a consistent recommendation dictionary."""
    if isinstance(movie, dict):
        title = movie.get('title') or movie.get('Title') or 'Unknown'
        genre = movie.get('genre') or movie.get('Genre') or 'Unknown'
        year = movie.get('year') or movie.get('Year') or ''
        normalized = dict(movie)
        normalized['title'] = title
        normalized['genre'] = genre
        normalized['year'] = year
        return normalized

    if isinstance(movie, (list, tuple)):
        return {
            'title': movie[1] if len(movie) > 1 else 'Unknown',
            'genre': movie[2] if len(movie) > 2 else 'Unknown',
            'year': movie[3] if len(movie) > 3 else '',
        }

    return {'title': str(movie), 'genre': 'Unknown', 'year': ''}


def build_feedback_profile(feedback_rows):
    """Aggregate feedback rows into reusable personalization signals.

    The latest vote for a title should win. Older votes are kept in the
    database for history, but they should not cancel or amplify the current
    preference when a user changes their mind.
    """
    if not feedback_rows:
        return {
            'title_scores': {},
            'genre_scores': {},
            'latest_votes': {},
            'liked_titles': [],
            'disliked_titles': [],
            'liked_genres': [],
            'disliked_genres': [],
        }

    title_scores = {}
    genre_scores = {}
    latest_votes = {}
    title_labels = {}
    processed_titles = set()

    for row in feedback_rows:
        if isinstance(row, dict):
            movie_title = row.get('movie_title') or row.get('title') or ''
            movie_genre = row.get('movie_genre') or row.get('genre') or ''
            vote_type = (row.get('vote_type') or '').strip().lower()
        else:
            movie_title = row[0] if len(row) > 0 else ''
            movie_genre = row[1] if len(row) > 1 else ''
            vote_type = (row[2] if len(row) > 2 else '').strip().lower()

        title_key = _normalize_title_key(movie_title)
        if not title_key or vote_type not in {'like', 'dislike'}:
            continue

        # Feedback rows are fetched newest-first from SQLite. Only the most
        # recent vote for each title should shape the active preference model.
        if title_key in processed_titles:
            continue
        processed_titles.add(title_key)

        delta = 1 if vote_type == 'like' else -1
        title_scores[title_key] = title_scores.get(title_key, 0) + (delta * 3)
        title_labels.setdefault(title_key, movie_title.strip())
        latest_votes.setdefault(title_key, vote_type)

        for genre_token in _extract_genre_tokens(movie_genre):
            genre_scores[genre_token] = genre_scores.get(genre_token, 0) + delta

    liked_titles = [
        title_labels[title_key]
        for title_key, score in sorted(title_scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ]
    disliked_titles = [
        title_labels[title_key]
        for title_key, score in sorted(title_scores.items(), key=lambda item: item[1])
        if score < 0
    ]
    liked_genres = [
        genre for genre, score in sorted(genre_scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ]
    disliked_genres = [
        genre for genre, score in sorted(genre_scores.items(), key=lambda item: item[1])
        if score < 0
    ]

    return {
        'title_scores': title_scores,
        'genre_scores': genre_scores,
        'latest_votes': latest_votes,
        'liked_titles': liked_titles,
        'disliked_titles': disliked_titles,
        'liked_genres': liked_genres,
        'disliked_genres': disliked_genres,
    }


def rank_recommendations_with_feedback(recommendations, feedback_profile=None, limit=None):
    """Re-rank recommendation candidates using saved likes and dislikes."""
    normalized_recommendations = [_normalize_movie_record(movie) for movie in recommendations]
    if not normalized_recommendations:
        return []

    if not feedback_profile:
        return normalized_recommendations[:limit] if limit else normalized_recommendations

    title_scores = feedback_profile.get('title_scores', {})
    genre_scores = feedback_profile.get('genre_scores', {})
    ranked_recommendations = []

    for index, movie in enumerate(normalized_recommendations):
        title_key = _normalize_title_key(movie.get('title'))
        feedback_bias = float(title_scores.get(title_key, 0))

        for genre_token in _extract_genre_tokens(movie.get('genre', '')):
            feedback_bias += genre_scores.get(genre_token, 0) * 0.8

        personalized_movie = dict(movie)
        personalized_movie['feedback_bias'] = round(feedback_bias, 2)
        personalized_movie['personalized_score'] = float(movie.get('similarity_score', 0) or 0) + feedback_bias
        ranked_recommendations.append((personalized_movie['personalized_score'], -index, personalized_movie))

    ranked_recommendations.sort(key=lambda item: (item[0], item[1]), reverse=True)
    personalized_results = [movie for _, _, movie in ranked_recommendations]
    return personalized_results[:limit] if limit else personalized_results


def _keyword_recommend(query, movies, top_k=5):
    """Fallback recommender when scikit-learn is unavailable."""
    query_terms = [
        term for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in TOKEN_STOPWORDS and len(term) > 2
    ]
    if not query_terms:
        return [_normalize_movie_record(movie) for movie in movies[:top_k]]

    scored_movies = []
    for movie in movies:
        normalized = _normalize_movie_record(movie)
        haystack = f"{normalized['title']} {normalized['genre']} {normalized['year']}".lower()
        score = 0
        for term in query_terms:
            if term in normalized['title'].lower():
                score += 3
            elif term in haystack:
                score += 1

        if score > 0:
            normalized['similarity_score'] = float(score)
            scored_movies.append((score, normalized))

    if not scored_movies:
        return [_normalize_movie_record(movie) for movie in movies[:top_k]]

    scored_movies.sort(key=lambda item: item[0], reverse=True)
    return [movie for _, movie in scored_movies[:top_k]]


def _extract_query_signals(query):
    lowered = query.lower()

    genres = [
        genre for genre, keywords in GENRE_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    themes = [
        theme for theme, keywords in THEME_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    moods = [
        mood for mood, profile in MOOD_PROFILES.items()
        if any(keyword in lowered for keyword in profile['keywords'])
    ]

    actors = []
    for pattern in ACTOR_PATTERNS:
        for match in re.findall(pattern, lowered):
            cleaned = re.sub(r"\b(movie|movies|film|films|anime|series)\b", "", match).strip(" ,.-")
            if len(cleaned.split()) >= 2:
                actors.append(cleaned.title())

    references = []
    for pattern in REFERENCE_TITLE_PATTERNS:
        for match in re.findall(pattern, query, flags=re.IGNORECASE):
            cleaned = match.strip(" ,.-")
            if cleaned:
                references.append(cleaned)

    return {
        'genres': genres,
        'themes': themes,
        'moods': moods,
        'actors': actors,
        'references': references,
    }


def _pick_value(movie, details, *keys):
    for key in keys:
        if isinstance(details, dict) and details.get(key):
            return details.get(key)
        if isinstance(movie, dict) and movie.get(key):
            return movie.get(key)
    return ""


def build_recommendation_reason(query, movie, details=None):
    """Generate a short, user-facing explanation for a recommendation."""
    normalized_movie = _normalize_movie_record(movie)
    signals = _extract_query_signals(query)

    title = normalized_movie.get('title', 'This pick')
    genre_text = _pick_value(normalized_movie, details, 'genre', 'Genre').lower()
    actors_text = _pick_value(normalized_movie, details, 'actors', 'Actors').lower()
    plot_text = _pick_value(normalized_movie, details, 'plot', 'Plot').lower()
    metadata_text = f"{title} {genre_text} {actors_text} {plot_text}".lower()

    actor_match = next(
        (actor for actor in signals['actors'] if actor.lower() in actors_text),
        None,
    )
    if actor_match:
        return f"Actor match: {actor_match} is part of the cast here."

    genre_match = next(
        (genre for genre in signals['genres'] if genre in genre_text or genre.replace('-', ' ') in genre_text),
        None,
    )
    if genre_match:
        return f"Genre match: it leans into the {genre_match.title()} style you asked for."

    theme_match = next(
        (
            theme for theme in signals['themes']
            if any(keyword in metadata_text for keyword in THEME_KEYWORDS[theme])
        ),
        None,
    )
    if theme_match:
        return f"Theme similarity: it taps into {theme_match.replace('-', ' ')} ideas from your prompt."

    mood_match = next(
        (
            mood for mood in signals['moods']
            if any(genre in genre_text for genre in MOOD_PROFILES[mood]['genres'])
        ),
        None,
    )
    if mood_match:
        descriptor = MOOD_PROFILES[mood_match]['descriptor']
        return f"Mood match: its {descriptor} tone fits the {mood_match} vibe you mentioned."

    if signals['references']:
        reference_title = signals['references'][0]
        primary_genre = normalized_movie.get('genre') or 'story'
        return f"Theme similarity: it carries a similar {primary_genre.lower()} energy to {reference_title}."

    if normalized_movie.get('genre') and normalized_movie['genre'] != 'Unknown':
        return f"Genre match: {normalized_movie['genre']} makes it a strong fit for this request."

    return "Theme similarity: the story setup lines up well with what you asked for."


def add_recommendation_reasons(query, recommendations, details_lookup=None):
    """Attach a recommendation_reason field to each recommendation."""
    enriched = []
    for recommendation in recommendations:
        normalized = _normalize_movie_record(recommendation)
        details = details_lookup(normalized) if callable(details_lookup) else None
        normalized['recommendation_reason'] = build_recommendation_reason(
            query,
            normalized,
            details,
        )
        enriched.append(normalized)
    return enriched

def tfidf_recommend(query, movies_df, top_k=5):
    """
    Recommend movies based on TF-IDF and cosine similarity
    Uses title, genre, and year for matching
    """
    if len(movies_df) == 0:
        return []
    
    # Convert DataFrame to list of dictionaries if it's not already
    if isinstance(movies_df, list):
        movies = movies_df
    else:
        movies = []
        for _, row in movies_df.iterrows():
            movies.append({
                'title': row['title'] if 'title' in row else row[1],
                'genre': row['genre'] if 'genre' in row else row[2],
                'year': row['year'] if 'year' in row else row[3]
            })
    
    if TfidfVectorizer is None or cosine_similarity is None:
        return _keyword_recommend(query, movies, top_k)

    # Create text features for each movie
    movie_texts = []
    for movie in movies:
        text = f"{movie['title']} {movie['genre']} {movie['year']}"
        movie_texts.append(text)
    
    # Add the query to the corpus
    all_texts = movie_texts + [query]
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between query and all movies
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Get top k most similar movies
    similar_indices = cosine_similarities.argsort()[0][-top_k:][::-1]
    
    recommendations = []
    for idx in similar_indices:
        if idx < len(movies):
            movie = movies[idx]
            movie['similarity_score'] = float(cosine_similarities[0][idx])
            recommendations.append(movie)
    
    return recommendations

def omdb_keyword_fallback(query, api_key=None, top_k=5):
    """
    Fallback to OMDb API for movie recommendations based on keywords
    This is used when the local database doesn't have enough matches
    """
    if not api_key:
        # Return some default movie suggestions if no API key
        return get_default_suggestions(query, top_k)
    
    try:
        # Search OMDb by keyword
        url = f"http://www.omdbapi.com/?s={query}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if data.get('Response') == 'True':
            movies = []
            for item in data['Search'][:top_k]:
                # Get detailed info for each movie
                detail_url = f"http://www.omdbapi.com/?i={item['imdbID']}&apikey={api_key}"
                detail_response = requests.get(detail_url)
                detail_data = detail_response.json()
                
                movie = {
                    'title': detail_data.get('Title', 'Unknown'),
                    'year': detail_data.get('Year', 'Unknown'),
                    'genre': detail_data.get('Genre', 'Unknown'),
                    'plot': detail_data.get('Plot', 'No description available'),
                    'type': detail_data.get('Type', 'movie')
                }
                movies.append(movie)
            
            return movies
        else:
            return get_default_suggestions(query, top_k)
            
    except Exception as e:
        print(f"OMDb API error: {e}")
        return get_default_suggestions(query, top_k)

def get_default_suggestions(query, top_k=5):
    """Provide default movie suggestions when API is unavailable"""
    # Sample movie database for fallback
    default_movies = [
        {
            'title': 'Inception',
            'year': '2010',
            'genre': 'Sci-Fi, Thriller',
            'plot': 'A thief who steals corporate secrets through dream-sharing technology.',
            'type': 'movie'
        },
        {
            'title': 'The Dark Knight',
            'year': '2008',
            'genre': 'Action, Crime, Drama',
            'plot': 'Batman faces the Joker, a criminal mastermind who seeks to undermine Batman.',
            'type': 'movie'
        },
        {
            'title': 'Interstellar',
            'year': '2014',
            'genre': 'Adventure, Drama, Sci-Fi',
            'plot': 'A team of explorers travel through a wormhole in space.',
            'type': 'movie'
        },
        {
            'title': 'The Shawshank Redemption',
            'year': '1994',
            'genre': 'Drama',
            'plot': 'Two imprisoned men bond over a number of years.',
            'type': 'movie'
        },
        {
            'title': 'Pulp Fiction',
            'year': '1994',
            'genre': 'Crime, Drama',
            'plot': 'Various interconnected stories of criminals in Los Angeles.',
            'type': 'movie'
        }
    ]
    
    # Return random sample based on query (simple keyword matching)
    query_lower = query.lower()
    filtered_movies = []
    
    for movie in default_movies:
        if (any(word in movie['genre'].lower() for word in query_lower.split()) or
            any(word in movie['title'].lower() for word in query_lower.split()) or
            'action' in query_lower and 'action' in movie['genre'].lower() or
            'comedy' in query_lower and 'comedy' in movie['genre'].lower() or
            'drama' in query_lower and 'drama' in movie['genre'].lower() or
            'sci' in query_lower and 'sci' in movie['genre'].lower()):
            filtered_movies.append(movie)
    
    if not filtered_movies:
        return random.sample(default_movies, min(top_k, len(default_movies)))
    
    return filtered_movies[:top_k]

def enrich_with_plots(movies, api_key=None):
    """
    Enrich movie data with plots from OMDb API
    If no API key, use default descriptions
    """
    if not api_key:
        # Add default plots if no API key
        for movie in movies:
            if 'plot' not in movie or not movie['plot']:
                movie['plot'] = f"A great {movie.get('genre', '')} movie from {movie.get('year', '')}."
        return movies
    
    try:
        for movie in movies:
            if 'plot' not in movie or not movie['plot']:
                # Try to get plot from OMDb
                url = f"http://www.omdbapi.com/?t={movie['title']}&y={movie.get('year', '')}&apikey={api_key}"
                response = requests.get(url)
                data = response.json()
                
                if data.get('Response') == 'True':
                    movie['plot'] = data.get('Plot', 'No description available')
                else:
                    movie['plot'] = f"A captivating {movie.get('genre', '')} story from {movie.get('year', '')}."
        
        return movies
        
    except Exception as e:
        print(f"Plot enrichment error: {e}")
        # Ensure all movies have at least a basic plot
        for movie in movies:
            if 'plot' not in movie or not movie['plot']:
                movie['plot'] = f"An engaging {movie.get('genre', '')} film from {movie.get('year', '')}."
        return movies

def recommend_similar_movies(movie_title, movies_df, top_k=5):
    """
    Recommend movies similar to a given movie title
    """
    return tfidf_recommend(movie_title, movies_df, top_k)
