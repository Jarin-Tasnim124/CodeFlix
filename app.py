import os
import sqlite3
from datetime import datetime, timedelta
import random
import requests
import json
import base64
import io
from PIL import Image
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from urllib.parse import quote
import logging
import threading
from functools import lru_cache, wraps
import hashlib
import html
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set

from recommender import (
    add_recommendation_reasons,
    build_feedback_profile,
    rank_recommendations_with_feedback,
    tfidf_recommend,
)

# Custom Exceptions
class MovieError(Exception):
    """Base exception for movie operations"""
    pass

class DatabaseError(MovieError):
    """Database operation failed"""
    pass

class APIError(MovieError):
    """External API call failed"""
    pass

# Import seed module
try:
    import seed
    # Test if the module loaded correctly
    print("[OK] seed.py imported successfully!")
except ImportError as e:
    print(f"[ERROR] Error importing seed: {e}")
    # Create a dummy module to prevent crashes
    class DummySeed:
        def get_sample_movies(self): 
            return []
        def seed_database(self, db_path): 
            print("Seed module not available - using fallback")
            return False
        def clear_database(self, db_path): 
            return False
    
    seed = DummySeed()

# -----------------------------
# Configuration and Logging
# -----------------------------
def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('codeflix.log')
        ]
    )

def get_config():
    """Get configuration from environment or secrets with fallback"""
    try:
        # Try to get from secrets, with fallback values
        omdb_api_key = st.secrets.get("OMDB_API_KEY", "9771bb71")
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")  # 🔑 NEW
        database_url = st.secrets.get("DATABASE_URL", "movies.db")
        debug_mode = st.secrets.get("DEBUG", "false").lower() == 'true'
        cache_timeout = int(st.secrets.get("CACHE_TIMEOUT", "300"))
        max_retries = int(st.secrets.get("MAX_RETRIES", "3"))
        
        return {
            'omdb_api_key': omdb_api_key,
            'gemini_api_key': gemini_api_key,   # 🔑 NEW
            'database_url': database_url,
            'debug_mode': debug_mode,
            'cache_timeout': cache_timeout,
            'max_retries': max_retries
        }
    except Exception as e:
        logging.warning(f"Could not load secrets, using default values: {e}")
        # Fallback configuration
        return {
            'omdb_api_key': "9771bb71",  # Default API key
            'gemini_api_key': "",        # 🔑 NEW
            'database_url': "movies.db",
            'debug_mode': False,
            'cache_timeout': 300,
            'max_retries': 3
        }

CONFIG = get_config()
setup_logging()

def check_environment():
    """Check if all required environment variables and secrets are available"""
    try:
        # Test if we can access secrets
        test_key = st.secrets.get("OMDB_API_KEY", None)
        
        if test_key is None:
            st.warning("⚠️ No secrets.toml file found. Using default configuration.")
            st.info("""
            For better experience, create a `.streamlit/secrets.toml` file with:
            
            ```toml
            OMDB_API_KEY = "your_omdb_api_key_here"
            DATABASE_URL = "movies.db"
            DEBUG = "false"
            CACHE_TIMEOUT = "300"
            MAX_RETRIES = "3"
            ```
            
            Get a free OMDB API key from: http://www.omdbapi.com/apikey.aspx
            """)
            return True  # Continue with default values
        
        # Test OMDB API key
        omdb = RateLimitedOMDbAPI()
        if not omdb.validate_api_key():
            st.warning("⚠️ OMDB API key appears to be invalid. Some features may not work properly.")
        
        return True
        
    except Exception as e:
        st.warning(f"⚠️ Configuration issue: {e}. Using default settings.")
        return True  # Continue with default values

# -----------------------------
# Gemini Movie Chat – ONE API KEY (GEMINI)
# -----------------------------
def call_gemini_api(prompt: str) -> str:
    """Call Gemini Flash 2.5 via REST API and return plain text."""
    api_key = CONFIG.get("gemini_api_key", "")
    if not api_key:
        return "❌ Gemini API key is missing. Please configure GEMINI_API_KEY in .streamlit/secrets.toml."

    # Gemini model that can use Google Search under the hood
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    params = {"key": api_key}

    try:
        resp = requests.post(url, params=params, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        # Read Gemini response
        if "candidates" in data and data["candidates"]:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            return text.strip() or "⚠ Gemini returned an empty reply."

        return "⚠ Gemini did not return any candidates."

    except requests.HTTPError as e:
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", {}).get("message", str(err_data))
        except Exception:
            err_msg = str(e)

        if resp.status_code == 429:
            return "⚠ The AI hit its usage limit (429). Please try again in a bit."
        return f"⚠ Gemini API error ({resp.status_code}): {err_msg}"
    except Exception as e:
        return f"⚠ Gemini request failed: {e}"


def parse_movies_from_response(answer: str):
    """
    Split the Gemini answer into:
    - clean_text: the explanation for the user
    - movie_titles: up to 3 movie titles parsed from the MOVIES: line
    """
    if "MOVIES:" not in answer:
        return answer.strip(), []

    before, after = answer.split("MOVIES:", 1)
    # Only look at the first line after "MOVIES:"
    first_line = after.strip().splitlines()[0] if after.strip() else ""
    parts = [p.strip() for p in first_line.split("|") if p.strip()]

    return before.strip(), parts[:3]


def build_movie_chat_prompt(user_message: str, feedback_context: str = "", filter_context: str = "") -> str:
    """
    Build a prompt so Gemini:
    - Understands mood/plot/actors/etc.
    - Uses Google search tools if needed
    - Returns answer + MOVIES line (max 3 titles).
    """
    return f"""
You are a friendly movie expert assistant inside an app called CodeFlix.

The user will type anything related to movies, for example:
- a plot description
- their current mood
- one or more actor names
- genre, language, year, etc.

Your job:
1. Understand what kind of movie(s) fit the request.
2. Think about 1–3 good, real movies that match.
3. Use your web knowledge / Google Search tools if you need newer or niche titles.
4. Answer in TWO parts:

PART 1 – Short answer (max 120 words)
Explain in very simple English why these movies fit the user request.

PART 2 – Movie list
On a NEW line, write exactly:
MOVIES: title1 | title2 | title3

Rules for MOVIES line:
- Between 1 and 3 titles.
- Only movie names, no year, no extra text, no emojis.
- If you really find nothing, write just: MOVIES:

User preference memory:
{feedback_context or "No saved likes or dislikes yet. Use only the current request."}

Treat this memory as a soft preference:
- Lean toward liked titles/genres when they fit.
- Avoid disliked titles/genres unless the user explicitly asks for them.

Active recommendation filters:
{filter_context or "No extra filters. You can recommend any matching movie or anime title."}

User message:
\"\"\"{user_message}\"\"\"
""".strip()


class GeminiMovieChat:
    """
    Gemini-only movie chatbot.
    ✅ Uses ONLY GEMINI_API_KEY.
    ❌ Does NOT call OMDb or any other movie API.
    """
    def __init__(self):
        self.history = []  # simple in-memory history (not sent back to Gemini for now)

    def generate_ai_response(self, user_message: str, feedback_context: str = "", filter_context: str = ""):
        prompt = build_movie_chat_prompt(
            user_message,
            feedback_context=feedback_context,
            filter_context=filter_context,
        )
        full_answer = call_gemini_api(prompt)

        # Split into explanation + up to 3 movie names
        answer_text, movie_titles = parse_movies_from_response(full_answer)

        # Store minimal history for UI if needed
        self.history.append(("user", user_message))
        self.history.append(("assistant", answer_text))

        return answer_text, movie_titles
        
# -----------------------------
# Application State Management
# -----------------------------
@dataclass
class AppState:
    """Centralized application state"""
    expanded_movies: Set[int] = field(default_factory=set)
    show_watch_options: Set[int] = field(default_factory=set)
    show_details: Set[int] = field(default_factory=set)
    show_rating: Set[int] = field(default_factory=set)
    current_page: int = 0
    filters: Dict = field(default_factory=dict)
    search_cache: Dict = field(default_factory=dict)

def init_session_state():
    """Initialize centralized session state"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    if 'advanced_ai' not in st.session_state:
        st.session_state.advanced_ai = AdvancedAIChat()
    default_ai_filters = get_ai_finder_filter_defaults()
    if 'ai_finder_filters' not in st.session_state:
        st.session_state.ai_finder_filters = dict(default_ai_filters)
    if 'ai_filter_year_range' not in st.session_state:
        st.session_state.ai_filter_year_range = default_ai_filters["year_range"]
    if 'ai_filter_min_imdb' not in st.session_state:
        st.session_state.ai_filter_min_imdb = default_ai_filters["min_imdb_rating"]
    if 'ai_filter_content_type' not in st.session_state:
        st.session_state.ai_filter_content_type = default_ai_filters["content_type"]

# -----------------------------
# Enhanced OMDB API Integration with Better Error Handling
# -----------------------------
class RateLimitedOMDbAPI:
    def __init__(self):
        self.api_key = CONFIG['omdb_api_key']
        self.base_url = "http://www.omdbapi.com/"
        self.last_call_time = 0
        self.min_interval = 1.0
        self.max_retries = CONFIG['max_retries']
        self.streaming_finder = EnhancedStreamingServiceFinder()
    
    def robust_omdb_call(self, params, max_retries=None):
        """More robust API call with comprehensive error handling"""
        if max_retries is None:
            max_retries = self.max_retries
            
        for attempt in range(max_retries):
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last_call = current_time - self.last_call_time
                
                if time_since_last_call < self.min_interval:
                    time.sleep(self.min_interval - time_since_last_call)
                
                self.last_call_time = time.time()
                
                # Add timeout and better error handling
                response = requests.get(self.base_url, params=params, timeout=15)
                
                if response.status_code != 200:
                    logging.warning(f"API returned status {response.status_code}, attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        # Fallback to sample data
                        return self._get_fallback_data(params.get("s", ""))
                    continue
                
                data = response.json()
                
                if data.get("Response") == "True":
                    return data
                else:
                    error_msg = data.get('Error', 'Unknown error')
                    logging.warning(f"OMDB API error: {error_msg}")
                    
                    # Fallback for empty results
                    if "not found" in error_msg.lower() or "movie not found" in error_msg.lower():
                        return self._get_fallback_data(params.get("s", ""))
                    
                    if attempt == max_retries - 1:
                        return self._get_fallback_data(params.get("s", ""))
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error(f"Failed to fetch from OMDB after {max_retries} attempts, using fallback")
                    return self._get_fallback_data(params.get("s", ""))
                time.sleep(2)  # Longer wait before retry
    
    def _get_fallback_data(self, query):
        """Provide fallback data when API fails"""
        if not query:
            return {"Response": "False", "Error": "No query provided"}
        
        # Return sample search results based on query
        fallback_movies = [
            {
                "Title": f"{query.title()} (Sample)",
                "Year": "2023",
                "imdbID": f"tt{random.randint(1000000, 9999999)}",
                "Type": "movie",
                "Poster": "N/A"
            }
            for _ in range(min(5, len(query)))
        ]
        
        return {
            "Response": "True",
            "Search": fallback_movies,
            "totalResults": str(len(fallback_movies))
        }
    
    def get_movie_details(self, title, year=None):
        """Get detailed movie information from OMDB API"""
        try:
            params = {
                "apikey": self.api_key,
                "t": title,
                "plot": "full",
                "r": "json"
            }
            if year:
                params["y"] = year
            
            data = self.robust_omdb_call(params)
            
            if data and data.get("Response") == "True":
                return data
            else:
                logging.warning(f"No details found for {title}")
                return self._create_fallback_details(title, year)
                
        except Exception as e:
            logging.error(f"Error in get_movie_details: {e}")
            return self._create_fallback_details(title, year)
    
    def _create_fallback_details(self, title, year):
        """Create fallback movie details"""
        return {
            "Title": title,
            "Year": str(year) if year else "2023",
            "Rated": "N/A",
            "Released": "N/A",
            "Runtime": "N/A",
            "Genre": "N/A",
            "Director": "N/A",
            "Writer": "N/A",
            "Actors": "N/A",
            "Plot": "No plot available.",
            "Language": "N/A",
            "Country": "N/A",
            "Awards": "N/A",
            "Poster": "N/A",
            "Ratings": [],
            "Metascore": "N/A",
            "imdbRating": "N/A",
            "imdbVotes": "N/A",
            "imdbID": f"tt{random.randint(1000000, 9999999)}",
            "Type": "movie",
            "DVD": "N/A",
            "BoxOffice": "N/A",
            "Production": "N/A",
            "Website": "N/A",
            "Response": "True"
        }
    
    def search_movies(self, query, search_type="movie", year=None):
        """Enhanced search with fallback mechanisms"""
        if not query or len(query.strip()) < 2:
            return []
        
        try:
            params = {
                "apikey": self.api_key,
                "s": query.strip(),
                "type": search_type,
                "r": "json"
            }
            if year:
                params["y"] = year
            
            data = self.robust_omdb_call(params)
            
            if data and "Search" in data:
                return data["Search"]
            else:
                # Enhanced fallback to local database
                return self._search_local_database(query, search_type)
                
        except Exception as e:
            logging.error(f"Error in enhanced movie search: {e}")
            return self._search_local_database(query, search_type)
    
    def _search_local_database(self, query, search_type):
        """Search local movie database as fallback"""
        query_lower = query.lower()
        results = []
        
        for movie in get_enhanced_movie_database():
            if query_lower in movie["title"].lower():
                results.append({
                    "Title": movie["title"],
                    "Year": str(movie["year"]),
                    "imdbID": f"tt{hash(movie['title']) % 10000000}",
                    "Type": "movie",
                    "Poster": "N/A",
                    "Genre": movie["genre"]
                })
        
        # Add some popular movies if no results found
        if not results and len(query) > 2:
            popular_matches = [
                movie for movie in get_enhanced_movie_database()
                if any(word in movie["title"].lower() for word in query_lower.split()[:2])
            ][:5]
            
            for movie in popular_matches:
                results.append({
                    "Title": movie["title"],
                    "Year": str(movie["year"]),
                    "imdbID": f"tt{hash(movie['title']) % 10000000}",
                    "Type": "movie",
                    "Poster": "N/A",
                    "Genre": movie["genre"]
                })
        
        return results[:8]  # Limit results

    def validate_api_key(self):
        """Validate OMDB API key"""
        try:
            test_result = self.get_movie_details("The Matrix")
            return test_result is not None and test_result.get("Response") == "True"
        except Exception:
            return False

    def get_movie_with_streaming_info(self, title, year=None):
        """Get movie details with streaming information"""
        movie_data = self.get_movie_details(title, year)
        if movie_data:
            # Add streaming availability
            movie_year = int(movie_data.get('Year', '2023')[:4]) if movie_data.get('Year') else 2023
            streaming_info = self.streaming_finder.get_watch_options(
                title, movie_year, movie_data.get('Genre', '')
            )
            movie_data['streaming_info'] = streaming_info
        return movie_data

# -----------------------------
# Enhanced Streaming & Ticketing Services
# -----------------------------
class StreamingServiceFinder:
    def __init__(self):
        self.streaming_services = {
            'netflix': {
                'name': 'Netflix',
                'color': '#E50914',
                'icon': '🎬',
                'base_url': 'https://www.netflix.com/search?q=',
                'availability_check': self.check_netflix_availability
            },
            'amazon': {
                'name': 'Amazon Prime',
                'color': '#00A8E1',
                'icon': '📦',
                'base_url': 'https://www.primevideo.com/search/ref=atv_nb_sr?phrase=',
                'availability_check': self.check_amazon_availability
            },
            'disney': {
                'name': 'Disney+',
                'color': '#113CCF',
                'icon': '🏰',
                'base_url': 'https://www.disneyplus.com/search/',
                'availability_check': self.check_disney_availability
            },
            'hbo': {
                'name': 'HBO Max',
                'color': '#821DE0',
                'icon': '📺',
                'base_url': 'https://play.hbomax.com/search?q=',
                'availability_check': self.check_hbo_availability
            },
            'hulu': {
                'name': 'Hulu',
                'color': '#1CE783',
                'icon': '🍿',
                'base_url': 'https://www.hulu.com/search?q=',
                'availability_check': self.check_hulu_availability
            },
            'apple': {
                'name': 'Apple TV+',
                'color': '#000000',
                'icon': '🍎',
                'base_url': 'https://tv.apple.com/search?term=',
                'availability_check': self.check_apple_availability
            },
            'youtube': {
                'name': 'YouTube Movies',
                'color': '#FF0000',
                'icon': '📹',
                'base_url': 'https://www.youtube.com/results?search_query=',
                'availability_check': self.check_youtube_availability
            },
            'crunchyroll': {
                'name': 'Crunchyroll',
                'color': '#F47521',
                'icon': '🍥',
                'base_url': 'https://www.crunchyroll.com/search?q=',
                'availability_check': self.check_crunchyroll_availability
            }
        }
        
        self.ticketing_services = {
            'fandango': {
                'name': 'Fandango',
                'color': '#FF7300',
                'icon': '🎟️',
                'base_url': 'https://www.fandango.com/search?q=',
                'type': 'tickets'
            },
            'amc': {
                'name': 'AMC Theatres',
                'color': '#FF0000',
                'icon': '🎭',
                'base_url': 'https://www.amctheatres.com/search?q=',
                'type': 'tickets'
            },
            'regal': {
                'name': 'Regal Cinemas',
                'color': '#FFD700',
                'icon': '👑',
                'base_url': 'https://www.regmovies.com/search?q=',
                'type': 'tickets'
            },
            'cinemark': {
                'name': 'Cinemark',
                'color': '#00A4E0',
                'icon': '💙',
                'base_url': 'https://www.cinemark.com/search?q=',
                'type': 'tickets'
            }
        }

    def check_netflix_availability(self, title, year):
        """Mock Netflix availability check"""
        netflix_titles = ["stranger things", "the crown", "wednesday", "squid game", "the irishman", "bird box", 
                         "the gray man", "red notice", "don't look up", "the adam project", "enola holmes"]
        return any(netflix_title in title.lower() for netflix_title in netflix_titles) or random.random() < 0.4

    def check_amazon_availability(self, title, year):
        """Mock Amazon Prime availability check"""
        amazon_titles = ["the boys", "the marvelous mrs. maisel", "jack ryan", "the terminal list", "reacher",
                        "the tomorrow war", "coming 2 america", "the boys in the boat"]
        return any(amazon_title in title.lower() for amazon_title in amazon_titles) or random.random() < 0.5

    def check_disney_availability(self, title, year):
        """Mock Disney+ availability check"""
        disney_titles = ["star wars", "marvel", "pixar", "national geographic", "the simpsons", "black panther",
                        "avatar", "guardians of the galaxy", "the little mermaid", "wish", "elemental"]
        return any(keyword in title.lower() for keyword in disney_titles) or random.random() < 0.3

    def check_hbo_availability(self, title, year):
        """Mock HBO Max availability check"""
        hbo_titles = ["game of thrones", "succession", "the last of us", "euphoria", "house of the dragon",
                     "dune", "the batman", "elvis", "shazam", "blue beetle", "the nun"]
        return any(keyword in title.lower() for keyword in hbo_titles) or random.random() < 0.35

    def check_hulu_availability(self, title, year):
        """Mock Hulu availability check"""
        return random.random() < 0.4

    def check_apple_availability(self, title, year):
        """Mock Apple TV+ availability check"""
        apple_titles = ["ted lasso", "see", "the morning show", "foundation", "for all mankind", "napoleon", "argylle"]
        return any(keyword in title.lower() for keyword in apple_titles) or random.random() < 0.25

    def check_youtube_availability(self, title, year):
        """Mock YouTube Movies availability check"""
        return random.random() < 0.6

    def check_crunchyroll_availability(self, title, year):
        """Mock Crunchyroll availability check"""
        crunchyroll_titles = ["demon slayer", "jujutsu kaisen", "attack on titan", "my hero academia", "one piece",
                            "naruto", "dragon ball", "death note", "hunter x hunter", "fullmetal alchemist", "spy x family",
                            "chainsaw man", "bleach", "one punch man", "tokyo revengers", "jojo's bizarre adventure"]
        return any(keyword in title.lower() for keyword in crunchyroll_titles) or random.random() < 0.7

    def get_streaming_availability(self, title, year):
        """Get streaming availability for a movie"""
        availability = {}
        
        for service_id, service in self.streaming_services.items():
            try:
                is_available = service['availability_check'](title, year)
                availability[service_id] = {
                    'name': service['name'],
                    'available': is_available,
                    'url': f"{service['base_url']}{quote(title)}",
                    'color': service['color'],
                    'icon': service['icon']
                }
            except Exception as e:
                logging.error(f"Error checking {service_id} availability: {e}")
                availability[service_id] = {
                    'name': service['name'],
                    'available': False,
                    'url': f"{service['base_url']}{quote(title)}",
                    'color': service['color'],
                    'icon': service['icon']
                }
        
        return availability

    def get_ticketing_links(self, title, year):
        """Get ticketing links for movies in theaters"""
        ticketing_links = {}
        
        for service_id, service in self.ticketing_services.items():
            ticketing_links[service_id] = {
                'name': service['name'],
                'url': f"{service['base_url']}{quote(title)}",
                'color': service['color'],
                'icon': service['icon'],
                'type': 'tickets'
            }
        
        return ticketing_links

    def is_movie_in_theaters(self, year):
        """Check if movie is likely in theaters (released in current or previous year)"""
        current_year = datetime.now().year
        return year >= current_year - 1

    def get_watch_options(self, title, year, genre):
        """Get all watch options including streaming and ticketing"""
        watch_options = {
            'streaming': self.get_streaming_availability(title, year),
            'ticketing': {},
            'in_theaters': self.is_movie_in_theaters(year)
        }
        
        if watch_options['in_theaters']:
            watch_options['ticketing'] = self.get_ticketing_links(title, year)
        
        return watch_options

# -----------------------------
# Anime Integration for Crunchyroll
# -----------------------------
class AnimeIntegration:
    def __init__(self):
        self.base_url = "https://www.crunchyroll.com"
        self.search_url = f"{self.base_url}/search"
        
    def search_anime(self, query):
        """Search for anime on Crunchyroll"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            params = {"q": query}
            response = requests.get(self.search_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return self._parse_anime_results(response.text, query)
            else:
                return self._get_fallback_anime_results(query)
                
        except Exception as e:
            logging.error(f"Error searching anime: {e}")
            return self._get_fallback_anime_results(query)
    
    def _parse_anime_results(self, html_content, query):
        """Parse anime results from HTML response"""
        return self._get_fallback_anime_results(query)
    
    def _get_fallback_anime_results(self, query):
        """Provide fallback anime data for Crunchyroll"""
        popular_anime = [
            {
                "title": "Demon Slayer: Kimetsu no Yaiba",
                "year": "2019",
                "genre": "Anime, Action, Supernatural",
                "url": f"{self.base_url}/series/GY5P48XEY/demon-slayer-kimetsu-no-yaiba",
                "type": "anime",
                "status": "Available",
                "service": "Crunchyroll"
            },
            {
                "title": "Jujutsu Kaisen",
                "year": "2020", 
                "genre": "Anime, Action, Supernatural",
                "url": f"{self.base_url}/series/GY5P48XEY/jujutsu-kaisen",
                "type": "anime",
                "status": "Available",
                "service": "Crunchyroll"
            },
            {
                "title": "Attack on Titan",
                "year": "2013",
                "genre": "Anime, Action, Drama",
                "url": f"{self.base_url}/series/GY5P48XEY/attack-on-titan", 
                "type": "anime",
                "status": "Available",
                "service": "Crunchyroll"
            },
            {
                "title": "My Hero Academia",
                "year": "2016",
                "genre": "Anime, Action, Superhero",
                "url": f"{self.base_url}/series/GY5P48XEY/my-hero-academia",
                "type": "anime", 
                "status": "Available",
                "service": "Crunchyroll"
            },
            {
                "title": "One Piece",
                "year": "1999",
                "genre": "Anime, Adventure, Action",
                "url": f"{self.base_url}/series/GY5P48XEY/one-piece",
                "type": "anime",
                "status": "Available",
                "service": "Crunchyroll"
            },
            {
                "title": "Chainsaw Man",
                "year": "2022",
                "genre": "Anime, Action, Supernatural",
                "url": f"{self.base_url}/series/GY5P48XEY/chainsaw-man",
                "type": "anime",
                "status": "Available",
                "service": "Crunchyroll"
            },
            {
                "title": "Spy x Family",
                "year": "2022",
                "genre": "Anime, Comedy, Action",
                "url": f"{self.base_url}/series/GY5P48XEY/spy-x-family",
                "type": "anime",
                "status": "Available",
                "service": "Crunchyroll"
            },
            {
                "title": "Hunter x Hunter",
                "year": "2011",
                "genre": "Anime, Adventure, Action",
                "url": f"{self.base_url}/series/GY5P48XEY/hunter-x-hunter",
                "type": "anime",
                "status": "Available",
                "service": "Crunchyroll"
            }
        ]
        
        # Filter by query if provided
        if query and query.strip():
            query_lower = query.lower()
            filtered_anime = [
                anime for anime in popular_anime 
                if query_lower in anime["title"].lower()
            ]
            return filtered_anime if filtered_anime else popular_anime[:3]
        
        return popular_anime

# -----------------------------
# Enhanced StreamingServiceFinder with Anime Support
# -----------------------------
class EnhancedStreamingServiceFinder(StreamingServiceFinder):
    def __init__(self):
        super().__init__()
        self.anime_integration = AnimeIntegration()
        
    def get_watch_options(self, title, year, genre):
        """Enhanced watch options with anime support"""
        watch_options = super().get_watch_options(title, year, genre)
        
        # Add anime streaming if it's anime content
        if self._is_anime_content(title, genre):
            anime_results = self.anime_integration.search_anime(title)
            if anime_results:
                watch_options['anime'] = {
                    'crunchyroll': {
                        'name': 'Crunchyroll',
                        'available': True,
                        'url': anime_results[0]['url'],
                        'color': '#F47521',
                        'icon': '🍥',
                        'type': 'anime'
                    }
                }
        
        return watch_options
    
    def _is_anime_content(self, title, genre):
        """Check if content is likely anime"""
        anime_keywords = ['anime', 'demon slayer', 'jujutsu kaisen', 'attack on titan', 
                         'my hero academia', 'one piece', 'naruto', 'dragon ball', 'death note',
                         'kimetsu no yaiba', 'shippuden', 'hunter x hunter', 'fullmetal alchemist',
                         'chainsaw man', 'spy x family', 'bleach', 'one punch man', 'jojo']
        
        title_lower = title.lower()
        genre_lower = genre.lower() if genre else ""
        
        return (any(keyword in title_lower for keyword in anime_keywords) or
                'anime' in genre_lower)

# -----------------------------
# Caching and Performance Optimizations
# -----------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_movie_details_cached(title: str, year: int = None) -> dict:
    """Cached movie details"""
    omdb = RateLimitedOMDbAPI()
    return omdb.get_movie_details(title, year)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_streaming_availability_cached(title: str, year: int, genre: str):
    """Cached streaming info"""
    streaming_finder = EnhancedStreamingServiceFinder()
    return streaming_finder.get_watch_options(title, year, genre)

@st.cache_data(ttl=60)  # Cache for 1 minute
def create_analytics_charts_cached(movie_data_hash: str):
    """Cache analytics charts using data hash"""
    movies_data = get_movies_safe()
    return create_advanced_analytics_charts(movies_data)

# Cached OMDB searches
@lru_cache(maxsize=100)
def cached_omdb_search(query, search_type):
    """Cache OMDB search results to reduce API calls"""
    omdb = RateLimitedOMDbAPI()
    return omdb.search_movies(query, search_type)

# -----------------------------
# Enhanced Movie Database with Real Streaming Content
# -----------------------------
def get_enhanced_movie_database():
    """Return enhanced movie database with real streaming content"""
    enhanced_db = [
        # Netflix Originals (Really streaming on Netflix)
        {"title": "The Gray Man", "genre": "Action", "year": 2022, "streaming_service": "Netflix"},
        {"title": "Red Notice", "genre": "Action", "year": 2021, "streaming_service": "Netflix"},
        {"title": "Don't Look Up", "genre": "Comedy", "year": 2021, "streaming_service": "Netflix"},
        {"title": "The Adam Project", "genre": "Sci-Fi", "year": 2022, "streaming_service": "Netflix"},
        {"title": "Enola Holmes 2", "genre": "Mystery", "year": 2022, "streaming_service": "Netflix"},
        {"title": "Glass Onion: A Knives Out Mystery", "genre": "Mystery", "year": 2022, "streaming_service": "Netflix"},
        {"title": "The Super Mario Bros. Movie", "genre": "Animation", "year": 2023, "streaming_service": "Netflix"},
        {"title": "Spider-Man: No Way Home", "genre": "Action", "year": 2021, "streaming_service": "Netflix"},
        {"title": "65", "genre": "Sci-Fi", "year": 2023, "streaming_service": "Netflix"},
        {"title": "No Hard Feelings", "genre": "Comedy", "year": 2023, "streaming_service": "Netflix"},
        {"title": "The Equalizer 3", "genre": "Action", "year": 2023, "streaming_service": "Netflix"},
        {"title": "Anyone But You", "genre": "Romance", "year": 2023, "streaming_service": "Netflix"},
        
        # Amazon Prime (Really streaming on Amazon)
        {"title": "The Tomorrow War", "genre": "Sci-Fi", "year": 2021, "streaming_service": "Amazon Prime"},
        {"title": "Coming 2 America", "genre": "Comedy", "year": 2021, "streaming_service": "Amazon Prime"},
        {"title": "The Boys in the Boat", "genre": "Drama", "year": 2023, "streaming_service": "Amazon Prime"},
        {"title": "Top Gun: Maverick", "genre": "Action", "year": 2022, "streaming_service": "Amazon Prime"},
        {"title": "Creed III", "genre": "Drama", "year": 2023, "streaming_service": "Amazon Prime"},
        {"title": "M3GAN", "genre": "Horror", "year": 2022, "streaming_service": "Amazon Prime"},
        {"title": "Fast X", "genre": "Action", "year": 2023, "streaming_service": "Amazon Prime"},
        {"title": "Talk to Me", "genre": "Horror", "year": 2022, "streaming_service": "Amazon Prime"},
        {"title": "The Beekeeper", "genre": "Action", "year": 2024, "streaming_service": "Amazon Prime"},
        
        # Disney+ (Really streaming on Disney+)
        {"title": "Black Panther: Wakanda Forever", "genre": "Action", "year": 2022, "streaming_service": "Disney+"},
        {"title": "Avatar: The Way of Water", "genre": "Sci-Fi", "year": 2022, "streaming_service": "Disney+"},
        {"title": "Guardians of the Galaxy Vol. 3", "genre": "Action", "year": 2023, "streaming_service": "Disney+"},
        {"title": "The Little Mermaid", "genre": "Adventure", "year": 2023, "streaming_service": "Disney+"},
        {"title": "Ant-Man and the Wasp: Quantumania", "genre": "Action", "year": 2023, "streaming_service": "Disney+"},
        {"title": "Elemental", "genre": "Animation", "year": 2023, "streaming_service": "Disney+"},
        {"title": "Wish", "genre": "Animation", "year": 2023, "streaming_service": "Disney+"},
        
        # HBO Max (Really streaming on HBO Max)
        {"title": "Dune", "genre": "Sci-Fi", "year": 2021, "streaming_service": "HBO Max"},
        {"title": "The Batman", "genre": "Action", "year": 2022, "streaming_service": "HBO Max"},
        {"title": "Elvis", "genre": "Drama", "year": 2022, "streaming_service": "HBO Max"},
        {"title": "Black Adam", "genre": "Action", "year": 2022, "streaming_service": "HBO Max"},
        {"title": "Shazam! Fury of the Gods", "genre": "Action", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "Evil Dead Rise", "genre": "Horror", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "The Flash", "genre": "Action", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "Blue Beetle", "genre": "Action", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "The Nun II", "genre": "Horror", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "Ferrari", "genre": "Drama", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "The Color Purple", "genre": "Musical", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "Aquaman and the Lost Kingdom", "genre": "Action", "year": 2023, "streaming_service": "HBO Max"},
        
        # Crunchyroll Anime
        {"title": "Demon Slayer: Kimetsu no Yaiba", "genre": "Anime", "year": 2019, "streaming_service": "Crunchyroll"},
        {"title": "Jujutsu Kaisen", "genre": "Anime", "year": 2020, "streaming_service": "Crunchyroll"},
        {"title": "Attack on Titan", "genre": "Anime", "year": 2013, "streaming_service": "Crunchyroll"},
        {"title": "My Hero Academia", "genre": "Anime", "year": 2016, "streaming_service": "Crunchyroll"},
        {"title": "One Piece", "genre": "Anime", "year": 1999, "streaming_service": "Crunchyroll"},
        {"title": "Chainsaw Man", "genre": "Anime", "year": 2022, "streaming_service": "Crunchyroll"},
        {"title": "Spy x Family", "genre": "Anime", "year": 2022, "streaming_service": "Crunchyroll"},
        {"title": "Hunter x Hunter", "genre": "Anime", "year": 2011, "streaming_service": "Crunchyroll"},
        {"title": "Naruto: Shippuden", "genre": "Anime", "year": 2007, "streaming_service": "Crunchyroll"},
        {"title": "Death Note", "genre": "Anime", "year": 2006, "streaming_service": "Crunchyroll"},
        
        # Current Theater Releases (2023-2024) - Really in theaters
        {"title": "Oppenheimer", "genre": "Drama", "year": 2023, "in_theaters": True},
        {"title": "Barbie", "genre": "Comedy", "year": 2023, "in_theaters": True},
        {"title": "Killers of the Flower Moon", "genre": "Drama", "year": 2023, "in_theaters": True},
        {"title": "The Marvels", "genre": "Action", "year": 2023, "in_theaters": True},
        {"title": "Wonka", "genre": "Adventure", "year": 2023, "in_theaters": True},
        {"title": "Aquaman and the Lost Kingdom", "genre": "Action", "year": 2023, "in_theaters": True},
        {"title": "Migration", "genre": "Animation", "year": 2023, "in_theaters": True},
        {"title": "Anyone But You", "genre": "Romance", "year": 2023, "in_theaters": True},
        {"title": "The Beekeeper", "genre": "Action", "year": 2024, "in_theaters": True},
        {"title": "Mean Girls", "genre": "Musical", "year": 2024, "in_theaters": True},
        {"title": "Argylle", "genre": "Action", "year": 2024, "in_theaters": True},
        {"title": "Lisa Frankenstein", "genre": "Comedy", "year": 2024, "in_theaters": True},
        {"title": "Bob Marley: One Love", "genre": "Biography", "year": 2024, "in_theaters": True},
        {"title": "Dune: Part Two", "genre": "Sci-Fi", "year": 2024, "in_theaters": True},
        {"title": "Ghostbusters: Frozen Empire", "genre": "Comedy", "year": 2024, "in_theaters": True},
        
        # Anime Movies
        {"title": "Demon Slayer: Kimetsu no Yaiba - To the Hashira Training", "genre": "Anime", "year": 2024, "in_theaters": True},
        {"title": "The Boy and the Heron", "genre": "Anime", "year": 2023, "streaming_service": "HBO Max"},
        {"title": "Suzume", "genre": "Anime", "year": 2022, "streaming_service": "Crunchyroll"},
        {"title": "Jujutsu Kaisen 0", "genre": "Anime", "year": 2021, "streaming_service": "Crunchyroll"},
        
        # Other Streaming Services
        {"title": "John Wick: Chapter 4", "genre": "Action", "year": 2023, "streaming_service": "Starz"},
        {"title": "Scream VI", "genre": "Horror", "year": 2023, "streaming_service": "Paramount+"},
        {"title": "Transformers: Rise of the Beasts", "genre": "Action", "year": 2023, "streaming_service": "Paramount+"},
        {"title": "Mission: Impossible - Dead Reckoning Part One", "genre": "Action", "year": 2023, "streaming_service": "Paramount+"},
        {"title": "Cocaine Bear", "genre": "Comedy", "year": 2023, "streaming_service": "Peacock"},
        {"title": "Five Nights at Freddy's", "genre": "Horror", "year": 2023, "streaming_service": "Peacock"},
        {"title": "The Hunger Games: The Ballad of Songbirds & Snakes", "genre": "Drama", "year": 2023, "streaming_service": "Starz"},
        {"title": "Napoleon", "genre": "Drama", "year": 2023, "streaming_service": "Apple TV+"},
        {"title": "Argylle", "genre": "Action", "year": 2024, "streaming_service": "Apple TV+"},
        {"title": "Mean Girls", "genre": "Musical", "year": 2024, "streaming_service": "Paramount+"},
        
        # Classic Movies
        {"title": "The Dark Knight", "genre": "Action", "year": 2008},
        {"title": "Inception", "genre": "Action", "year": 2010},
        {"title": "The Shawshank Redemption", "genre": "Drama", "year": 1994},
        {"title": "Pulp Fiction", "genre": "Crime", "year": 1994},
        {"title": "Forrest Gump", "genre": "Drama", "year": 1994},
        {"title": "The Godfather", "genre": "Crime", "year": 1972},
        {"title": "The Matrix", "genre": "Sci-Fi", "year": 1999},
        {"title": "Interstellar", "genre": "Sci-Fi", "year": 2014},
    ]
    
    return enhanced_db

MOVIE_DATABASE = get_enhanced_movie_database()


def find_catalog_movie(title):
    """Find the closest local catalog match for a recommended title."""
    normalized_title = (title or "").strip().lower()
    if not normalized_title:
        return {"title": "Unknown", "genre": "Unknown", "year": ""}

    exact_match = next(
        (movie for movie in MOVIE_DATABASE if movie["title"].strip().lower() == normalized_title),
        None,
    )
    if exact_match:
        return dict(exact_match)

    loose_match = next(
        (
            movie for movie in MOVIE_DATABASE
            if normalized_title in movie["title"].lower() or movie["title"].lower() in normalized_title
        ),
        None,
    )
    if loose_match:
        return dict(loose_match)

    return {"title": title, "genre": "Unknown", "year": ""}


def resolve_recommendation_details(movie, include_remote_details=False):
    """Combine local recommendation metadata with optional OMDb details."""
    resolved_movie = dict(movie)
    details = {
        "Title": resolved_movie.get("title", ""),
        "Genre": resolved_movie.get("genre", "Unknown"),
        "Year": str(resolved_movie.get("year", "")),
    }

    if not include_remote_details:
        return resolved_movie, details

    genre = str(resolved_movie.get("genre", "")).lower()
    if "anime" in genre:
        return resolved_movie, details

    year_value = resolved_movie.get("year")
    parsed_year = None
    if isinstance(year_value, int):
        parsed_year = year_value
    elif isinstance(year_value, str):
        digits = "".join(ch for ch in year_value if ch.isdigit())
        if digits:
            parsed_year = int(digits[:4])

    remote_details = get_movie_details_cached(resolved_movie["title"], parsed_year)
    if remote_details:
        details.update(remote_details)
        if remote_details.get("Title") and remote_details["Title"] != "N/A":
            resolved_movie["title"] = remote_details["Title"]
        if remote_details.get("Genre") and remote_details["Genre"] != "N/A":
            resolved_movie["genre"] = remote_details["Genre"]
        if remote_details.get("Year") and remote_details["Year"] != "N/A":
            resolved_movie["year"] = remote_details["Year"]

    return resolved_movie, details


def build_explainable_recommendations(query, recommendations, include_remote_details=False):
    """Return recommendation cards with short explanation text."""
    normalized_recommendations = []
    details_cache = {}

    for recommendation in recommendations:
        if isinstance(recommendation, str):
            recommendation_movie = find_catalog_movie(recommendation)
        else:
            recommendation_movie = {
                **recommendation,
                "title": recommendation.get("title") or recommendation.get("Title") or "Unknown",
                "genre": recommendation.get("genre") or recommendation.get("Genre") or "Unknown",
                "year": recommendation.get("year") or recommendation.get("Year") or "",
            }

        resolved_movie, details = resolve_recommendation_details(
            recommendation_movie,
            include_remote_details=include_remote_details,
        )
        cache_key = f"{resolved_movie.get('title', '')}|{resolved_movie.get('year', '')}"
        details_cache[cache_key] = details
        normalized_recommendations.append(resolved_movie)

    def lookup_details(movie):
        cache_key = f"{movie.get('title', '')}|{movie.get('year', '')}"
        return details_cache.get(cache_key, {})

    return add_recommendation_reasons(
        query,
        normalized_recommendations,
        details_lookup=lookup_details,
    )


def normalize_feedback_title(title):
    """Normalize movie titles for feedback lookups."""
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def parse_year_value(year_value):
    """Extract a 4-digit year when possible."""
    if isinstance(year_value, int):
        return year_value
    if isinstance(year_value, str):
        digits = "".join(ch for ch in year_value if ch.isdigit())
        if digits:
            return int(digits[:4])
    return None


def parse_imdb_rating_value(rating_value):
    """Convert IMDB rating text into a float."""
    if rating_value in (None, "", "N/A"):
        return None
    try:
        return float(str(rating_value).split("/")[0].strip())
    except (TypeError, ValueError):
        return None


def detect_recommendation_content_type(movie):
    """Classify a recommendation as movie or anime."""
    genre_text = str(movie.get("genre") or movie.get("Genre") or "").lower()
    title_text = str(movie.get("title") or movie.get("Title") or "").lower()
    return "Anime" if "anime" in genre_text or "anime" in title_text else "Movie"


def get_ai_finder_filter_defaults():
    """Default filter values for the AI Finder recommender."""
    catalog_years = [
        parse_year_value(movie.get("year"))
        for movie in MOVIE_DATABASE
        if parse_year_value(movie.get("year")) is not None
    ]
    current_year = datetime.now().year
    min_year = min(catalog_years) if catalog_years else 1980
    max_year = max(max(catalog_years), current_year) if catalog_years else current_year
    return {
        "year_range": (min_year, max_year),
        "min_imdb_rating": 0.0,
        "content_type": "All",
    }


def reset_ai_finder_filters():
    """Reset AI Finder filter widgets to their default values."""
    default_filters = get_ai_finder_filter_defaults()
    st.session_state.ai_filter_year_range = default_filters["year_range"]
    st.session_state.ai_filter_min_imdb = default_filters["min_imdb_rating"]
    st.session_state.ai_filter_content_type = default_filters["content_type"]
    st.session_state.ai_finder_filters = default_filters


def build_ai_filter_summary(filters):
    """Human-readable summary of active recommendation filters."""
    if not filters:
        return ""

    defaults = get_ai_finder_filter_defaults()
    summary_parts = []
    year_range = filters.get("year_range")
    if year_range and tuple(year_range) != tuple(defaults["year_range"]):
        summary_parts.append(f"Years {year_range[0]}-{year_range[1]}")
    min_imdb_rating = float(filters.get("min_imdb_rating", 0.0) or 0.0)
    if min_imdb_rating > 0:
        summary_parts.append(f"IMDb >= {min_imdb_rating:.1f}")
    content_type = filters.get("content_type", "All")
    if content_type != "All":
        summary_parts.append(content_type)

    return "Recommendation filters: " + " | ".join(summary_parts) if summary_parts else ""


def build_ai_filter_prompt_context(filters):
    """Translate UI filters into prompt instructions for Gemini."""
    if not filters:
        return ""

    defaults = get_ai_finder_filter_defaults()
    instructions = []
    year_range = filters.get("year_range")
    if year_range and tuple(year_range) != tuple(defaults["year_range"]):
        instructions.append(f"Only recommend titles released between {year_range[0]} and {year_range[1]}.")
    min_imdb_rating = float(filters.get("min_imdb_rating", 0.0) or 0.0)
    if min_imdb_rating > 0:
        instructions.append(f"Prefer titles with IMDb rating {min_imdb_rating:.1f} or higher.")
    content_type = filters.get("content_type", "All")
    if content_type == "Movie":
        instructions.append("Only recommend movies, not anime series.")
    elif content_type == "Anime":
        instructions.append("Only recommend anime titles.")

    return " ".join(instructions)


def recommendation_matches_basic_filters(movie, filters):
    """Apply the filters that can be evaluated without remote rating lookups."""
    if not filters:
        return True

    content_type = filters.get("content_type", "All")
    detected_type = detect_recommendation_content_type(movie)
    if content_type != "All" and detected_type != content_type:
        return False

    parsed_year = parse_year_value(movie.get("year") or movie.get("Year"))
    year_range = filters.get("year_range")
    if year_range:
        if parsed_year is None:
            return False
        if parsed_year < year_range[0] or parsed_year > year_range[1]:
            return False

    return True


def recommendation_matches_filters(movie, details, filters):
    """Apply AI Finder filters to a single recommendation."""
    if not recommendation_matches_basic_filters(movie, filters):
        return False

    min_imdb_rating = float(filters.get("min_imdb_rating", 0.0) or 0.0)
    if min_imdb_rating > 0:
        rating_value = parse_imdb_rating_value(details.get("imdbRating") or details.get("details_rating"))
        if rating_value is None or rating_value < min_imdb_rating:
            return False

    return True


def filter_recommendations_for_ai_finder(recommendations, filters, include_remote_details=False):
    """Filter recommendation cards by year, IMDb rating, and content type."""
    if not recommendations:
        return []

    filtered_recommendations = []
    needs_rating_lookup = include_remote_details and float(filters.get("min_imdb_rating", 0.0) or 0.0) > 0

    for recommendation in recommendations:
        if isinstance(recommendation, str):
            recommendation = find_catalog_movie(recommendation)
        resolved_movie, details = resolve_recommendation_details(
            recommendation,
            include_remote_details=include_remote_details or needs_rating_lookup,
        )
        if recommendation_matches_filters(resolved_movie, details, filters):
            enriched_movie = dict(resolved_movie)
            imdb_rating = details.get("imdbRating") or details.get("details_rating")
            if imdb_rating and imdb_rating != "N/A":
                enriched_movie["imdb_rating"] = imdb_rating
            enriched_movie["content_type"] = detect_recommendation_content_type(enriched_movie)
            filtered_recommendations.append(enriched_movie)

    return filtered_recommendations


def get_recommendation_feedback_rows(limit=250):
    """Fetch recent recommendation feedback records."""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(
                """
                SELECT movie_title, movie_genre, movie_year, vote_type, source_query, created_at
                FROM recommendation_feedback
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in c.fetchall()]
    except sqlite3.Error as e:
        logging.error(f"Database error in get_recommendation_feedback_rows: {e}")
        return []


def get_recommendation_feedback_context(limit=250):
    """Return the aggregated feedback profile and the latest vote per title."""
    feedback_rows = get_recommendation_feedback_rows(limit=limit)
    latest_votes = {}

    for row in feedback_rows:
        title_key = normalize_feedback_title(row.get("movie_title"))
        vote_type = row.get("vote_type")
        if title_key and vote_type and title_key not in latest_votes:
            latest_votes[title_key] = vote_type

    return build_feedback_profile(feedback_rows), latest_votes


def save_recommendation_feedback(movie, vote_type, source_query=""):
    """Persist like or dislike feedback for a recommendation."""
    title = (movie.get("title") or movie.get("Title") or "").strip()
    if not title or vote_type not in {"like", "dislike"}:
        return None

    genre = movie.get("genre") or movie.get("Genre") or "Unknown"
    year_value = movie.get("year") or movie.get("Year") or ""
    year_text = str(year_value).strip()

    def operation(c, title, genre, year_text, vote_type, source_query):
        c.execute(
            """
            INSERT INTO recommendation_feedback (movie_title, movie_genre, movie_year, vote_type, source_query)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, genre, year_text, vote_type, source_query),
        )
        return c.lastrowid

    return safe_db_operation(operation, title, genre, year_text, vote_type, source_query)


def build_feedback_prompt_context(feedback_profile):
    """Summarize saved preferences for the Gemini prompt."""
    if not feedback_profile:
        return ""

    context_parts = []
    liked_titles = feedback_profile.get("liked_titles", [])[:3]
    disliked_titles = feedback_profile.get("disliked_titles", [])[:3]
    liked_genres = [genre.title() for genre in feedback_profile.get("liked_genres", [])[:3]]
    disliked_genres = [genre.title() for genre in feedback_profile.get("disliked_genres", [])[:3]]

    if liked_titles:
        context_parts.append(f"Liked titles: {', '.join(liked_titles)}.")
    if liked_genres:
        context_parts.append(f"Liked genres: {', '.join(liked_genres)}.")
    if disliked_titles:
        context_parts.append(f"Disliked titles: {', '.join(disliked_titles)}.")
    if disliked_genres:
        context_parts.append(f"Disliked genres: {', '.join(disliked_genres)}.")

    return " ".join(context_parts)


def build_feedback_summary_text(feedback_profile):
    """Small UI summary of the current personalization memory."""
    if not feedback_profile:
        return ""

    summary_parts = []
    liked_genres = [genre.title() for genre in feedback_profile.get("liked_genres", [])[:2]]
    disliked_genres = [genre.title() for genre in feedback_profile.get("disliked_genres", [])[:2]]

    if liked_genres:
        summary_parts.append(f"likes {', '.join(liked_genres)}")
    if disliked_genres:
        summary_parts.append(f"avoids {', '.join(disliked_genres)}")

    return "Personalization active: " + " | ".join(summary_parts) if summary_parts else ""


def build_feedback_aware_recommendations(query, recommendations, include_remote_details=False, limit=3, filters=None):
    """Blend AI picks with local candidates, then rank them using saved feedback."""
    feedback_profile, _ = get_recommendation_feedback_context()
    catalog_pool = [movie for movie in MOVIE_DATABASE if recommendation_matches_basic_filters(movie, filters)]
    if not catalog_pool:
        catalog_pool = list(MOVIE_DATABASE)

    candidate_pool = list(recommendations or [])
    candidate_pool.extend(tfidf_recommend(query, catalog_pool, top_k=max(limit * 4, 12)))

    for liked_genre in feedback_profile.get("liked_genres", [])[:2]:
        genre_matches = [
            movie for movie in catalog_pool
            if liked_genre in str(movie.get("genre", "")).lower()
        ]
        candidate_pool.extend(genre_matches[:3])

    deduped_candidates = []
    seen_titles = set()
    for candidate in candidate_pool:
        if isinstance(candidate, str):
            title = candidate
        else:
            title = candidate.get("title") or candidate.get("Title") or ""

        title_key = normalize_feedback_title(title)
        if not title_key or title_key in seen_titles:
            continue

        seen_titles.add(title_key)
        deduped_candidates.append(candidate)

    explained_recommendations = build_explainable_recommendations(
        query,
        deduped_candidates,
        include_remote_details=include_remote_details,
    )
    filtered_recommendations = filter_recommendations_for_ai_finder(
        explained_recommendations,
        filters or get_ai_finder_filter_defaults(),
        include_remote_details=include_remote_details,
    )
    ranked_recommendations = rank_recommendations_with_feedback(
        filtered_recommendations,
        feedback_profile=feedback_profile,
        limit=limit,
    )

    return ranked_recommendations or filtered_recommendations[:limit], feedback_profile


def build_quick_action_prompt(action_key, context_hint=""):
    """Create ready-to-run AI Finder prompts for quick action buttons."""
    cleaned_hint = (context_hint or "").strip()
    if action_key == "mood":
        return "I want a feel-good movie night. Recommend 3 movies based on a happy mood."
    if action_key == "similar":
        reference_title = cleaned_hint or "Inception"
        return f"Recommend 3 movies similar to {reference_title} and explain the match simply."
    if action_key == "anime":
        if cleaned_hint:
            return f"Recommend 3 anime-only titles similar in vibe to {cleaned_hint}."
        return "Recommend 3 anime-only titles with a strong story and memorable characters."
    return ""


def should_use_smart_fallback(answer_text, movie_titles):
    """Detect when the AI Finder should switch to the local TF-IDF recommender."""
    normalized_answer = (answer_text or "").strip().lower()
    failure_markers = [
        "gemini api key is missing",
        "gemini request failed",
        "gemini api error",
        "gemini did not return any candidates",
        "gemini returned an empty reply",
        "usage limit (429)",
        "hit its usage limit",
    ]

    if not normalized_answer and not movie_titles:
        return True, "empty_ai_reply"
    if any(marker in normalized_answer for marker in failure_markers):
        return True, "gemini_unavailable"
    if not movie_titles:
        return True, "no_ai_titles"

    return False, ""


def build_smart_fallback_recommendations(query, feedback_profile, limit=3, filters=None):
    """Generate local TF-IDF recommendations without depending on Gemini or OMDb."""
    catalog_pool = [movie for movie in MOVIE_DATABASE if recommendation_matches_basic_filters(movie, filters)]
    if not catalog_pool:
        catalog_pool = list(MOVIE_DATABASE)

    tfidf_candidates = tfidf_recommend(query, catalog_pool, top_k=max(limit * 4, 12))
    explained_candidates = build_explainable_recommendations(
        query,
        tfidf_candidates,
        include_remote_details=False,
    )
    filtered_candidates = filter_recommendations_for_ai_finder(
        explained_candidates,
        filters or get_ai_finder_filter_defaults(),
        include_remote_details=False,
    )
    ranked_candidates = rank_recommendations_with_feedback(
        filtered_candidates,
        feedback_profile=feedback_profile,
        limit=limit,
    )

    return ranked_candidates or filtered_candidates[:limit]


def build_smart_fallback_message(query, recommendations, fallback_reason, filters=None):
    """Create friendly UI text for local recommender fallback mode."""
    filter_summary = build_ai_filter_summary(filters or {})
    if recommendations:
        if fallback_reason == "gemini_unavailable":
            message = (
                "Live AI recommendations are unavailable right now, so I switched to the local "
                "TF-IDF recommender and picked the closest matches from your request."
            )
        else:
            message = (
            "I could not get reliable title picks from the live AI response, so I used the local "
            "TF-IDF recommender to find movies that match your prompt."
            )
        if filter_summary:
            message += f" {filter_summary}."
        return message

    no_match_message = (
        f"I could not find strong matches for '{query}' from Gemini or the local recommender yet. "
        "Try adding a genre, mood, actor, or a reference movie."
    )
    if filter_summary:
        no_match_message += f" Current filters may be too strict: {filter_summary.lower()}."
    return no_match_message


def build_ai_finder_assistant_message(chat, user_msg, feedback_profile, recommendation_filters=None):
    """Generate the assistant payload for one AI Finder prompt."""
    feedback_context = build_feedback_prompt_context(feedback_profile)
    filter_context = build_ai_filter_prompt_context(recommendation_filters or {})
    answer_text, movie_titles = chat.generate_ai_response(
        user_msg,
        feedback_context=feedback_context,
        filter_context=filter_context,
    )
    use_fallback, fallback_reason = should_use_smart_fallback(answer_text, movie_titles)

    if use_fallback:
        explained_recommendations = build_smart_fallback_recommendations(
            user_msg,
            feedback_profile,
            limit=3,
            filters=recommendation_filters,
        )
        answer_text = build_smart_fallback_message(
            user_msg,
            explained_recommendations,
            fallback_reason,
            filters=recommendation_filters,
        )
        updated_feedback_profile = get_recommendation_feedback_context()[0]
    else:
        explained_recommendations, updated_feedback_profile = build_feedback_aware_recommendations(
            user_msg,
            movie_titles,
            include_remote_details=True,
            filters=recommendation_filters,
        )
        if not explained_recommendations:
            answer_text = (
                "I found possible matches, but none of them satisfied your current filters. "
                "Try widening the year range, lowering the IMDb threshold, or switching content type."
            )

    assistant_message = {
        "role": "assistant",
        "content": answer_text,
        "recommendations": explained_recommendations,
        "source_query": user_msg,
        "used_fallback": use_fallback,
        "filters": recommendation_filters or {},
    }
    return assistant_message, updated_feedback_profile


def sync_ai_finder_chat_history(chat):
    """Keep the lightweight Gemini history aligned with the visible chat."""
    if not chat:
        return

    chat.history = [
        (message.get("role", ""), message.get("content", ""))
        for message in st.session_state.get("ai_history", [])
        if message.get("role") in {"user", "assistant"}
    ]


def get_ai_finder_history_turns():
    """Convert flat AI Finder history into user/assistant turns."""
    turns = []
    history = st.session_state.get("ai_history", [])
    index = 0

    while index < len(history):
        message = history[index]
        if message.get("role") != "user":
            index += 1
            continue

        assistant_message = None
        assistant_index = None
        if index + 1 < len(history) and history[index + 1].get("role") == "assistant":
            assistant_index = index + 1
            assistant_message = history[assistant_index]

        turns.append(
            {
                "user_index": index,
                "assistant_index": assistant_index,
                "user_message": message,
                "assistant_message": assistant_message,
            }
        )
        index = assistant_index + 1 if assistant_index is not None else index + 1

    return turns


def clear_ai_finder_edit_state():
    """Reset temporary UI state for AI Finder prompt editing."""
    st.session_state.ai_edit_turn_index = None
    st.session_state.ai_edit_prompt = ""


def delete_ai_finder_history_turn(chat, user_index):
    """Delete a prompt and its paired assistant response from AI Finder history."""
    history = list(st.session_state.get("ai_history", []))
    if user_index < 0 or user_index >= len(history):
        return False
    if history[user_index].get("role") != "user":
        return False

    delete_count = 1
    if user_index + 1 < len(history) and history[user_index + 1].get("role") == "assistant":
        delete_count = 2

    del history[user_index:user_index + delete_count]
    st.session_state.ai_history = history
    clear_ai_finder_edit_state()
    sync_ai_finder_chat_history(chat)
    return True


def update_ai_finder_history_turn(chat, user_index, user_msg, feedback_profile, recommendation_filters=None):
    """Edit a saved prompt and regenerate its paired assistant message."""
    history = list(st.session_state.get("ai_history", []))
    if user_index < 0 or user_index >= len(history):
        return None, feedback_profile
    if history[user_index].get("role") != "user":
        return None, feedback_profile

    clean_prompt = user_msg.strip()
    if not clean_prompt:
        return None, feedback_profile

    assistant_message, updated_feedback_profile = build_ai_finder_assistant_message(
        chat,
        clean_prompt,
        feedback_profile,
        recommendation_filters=recommendation_filters,
    )

    history[user_index] = {"role": "user", "content": clean_prompt}
    if user_index + 1 < len(history) and history[user_index + 1].get("role") == "assistant":
        history[user_index + 1] = assistant_message
    else:
        history.insert(user_index + 1, assistant_message)

    st.session_state.ai_history = history
    clear_ai_finder_edit_state()
    sync_ai_finder_chat_history(chat)
    return assistant_message, updated_feedback_profile


def run_ai_finder_chat_turn(chat, user_msg, feedback_profile, recommendation_filters=None):
    """Process a chat prompt and persist the full exchange into session history."""
    clear_ai_finder_edit_state()
    st.session_state.ai_history.append({"role": "user", "content": user_msg})
    assistant_message, updated_feedback_profile = build_ai_finder_assistant_message(
        chat,
        user_msg,
        feedback_profile,
        recommendation_filters=recommendation_filters,
    )
    st.session_state.ai_history.append(assistant_message)
    sync_ai_finder_chat_history(chat)

    return assistant_message, updated_feedback_profile


def render_explainable_recommendations(
    recommendations,
    key_prefix,
    show_add_button=False,
    feedback_states=None,
    feedback_query="",
):
    """Render recommendation cards with explanation text."""
    if feedback_states is None:
        _, feedback_states = get_recommendation_feedback_context()

    for index, movie in enumerate(recommendations):
        year = movie.get("year", "")
        year_label = f" ({year})" if year else ""
        st.markdown(f"**{movie.get('title', 'Unknown')}**{year_label}")

        genre = movie.get("genre")
        if genre and genre != "Unknown":
            st.caption(genre)

        metadata_bits = []
        if movie.get("imdb_rating"):
            metadata_bits.append(f"IMDb {movie['imdb_rating']}")
        if movie.get("content_type") and movie.get("content_type") != "Movie":
            metadata_bits.append(movie["content_type"])
        if metadata_bits:
            st.caption(" | ".join(metadata_bits))

        st.markdown(
            f"**Why this movie?** {movie.get('recommendation_reason', 'It matches the story vibe in your request.')}"
        )

        current_vote = feedback_states.get(normalize_feedback_title(movie.get("title", "")))
        control_columns = st.columns([1, 1, 1] if show_add_button else [1, 1])

        with control_columns[0]:
            if st.button(
                "👍 Liked" if current_vote == "like" else "👍",
                key=f"{key_prefix}_like_{index}",
                use_container_width=True,
                type="primary" if current_vote == "like" else "secondary",
            ):
                save_recommendation_feedback(movie, "like", feedback_query)
                st.session_state.feedback_flash = f"Saved a like for {movie['title']}."
                st.rerun()

        with control_columns[1]:
            if st.button(
                "👎 Disliked" if current_vote == "dislike" else "👎",
                key=f"{key_prefix}_dislike_{index}",
                use_container_width=True,
                type="primary" if current_vote == "dislike" else "secondary",
            ):
                save_recommendation_feedback(movie, "dislike", feedback_query)
                st.session_state.feedback_flash = f"Saved a dislike for {movie['title']}."
                st.rerun()

        if show_add_button:
            with control_columns[2]:
                if st.button(
                    "Add",
                    key=f"{key_prefix}_{index}",
                    use_container_width=True,
                    type="primary",
                ):
                    year_value = movie.get("year")
                    if isinstance(year_value, str):
                        digits = "".join(ch for ch in year_value if ch.isdigit())
                        year_value = int(digits[:4]) if digits else 2023
                    elif not isinstance(year_value, int):
                        year_value = 2023

                    add_movie(movie["title"], movie.get("genre", "Unknown"), year_value, False)
                    st.success(f"Added {movie['title']}!")

        # Comment section below every recommended movie
        render_movie_comments_section(
            movie_title=movie.get("title", f"movie_{index}"),
            key_prefix=f"{key_prefix}_cmt_{index}",
        )

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
        flash_message = st.session_state.pop(flash_key, None)
        if flash_message:
            if isinstance(flash_message, tuple):
                flash_kind, flash_text = flash_message
            else:
                flash_kind, flash_text = "success", flash_message

            if flash_kind == "warning":
                st.warning(flash_text)
            elif flash_kind == "error":
                st.error(flash_text)
            else:
                st.success(flash_text)

        # existing comments
        if comments:
            for cmt in comments:
                cmt_id   = cmt["id"]
                cmt_text = cmt["text"]
                ts       = cmt["created_at"][:16] if cmt["created_at"] else ""
                edited   = " (edited)" if cmt["updated_at"] != cmt["created_at"] else ""

                edit_flag_key = f"cmnt_edit_{widget_key}_{cmt_id}"
                edit_text_key = f"cmnt_etext_{widget_key}_{cmt_id}"

                # styled comment card — text + timestamp + inline action row
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.05);border-left:3px solid #FFD93D;
                                padding:0.6rem 0.8rem 0.5rem;border-radius:6px;margin-bottom:0.3rem;">
                        <div style="color:#eee;font-size:0.93rem;margin-bottom:0.35rem;">
                            {html.escape(cmt_text)}
                        </div>
                        <div style="color:#888;font-size:0.70rem;">{ts}{edited}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Edit / Delete — perfectly equal, flush side-by-side
                edit_col, del_col = st.columns([1, 1])
                with edit_col:
                    if st.button("✏️ Edit", key=f"btn_edit_{widget_key}_{cmt_id}",
                                 use_container_width=True):
                        st.session_state[edit_flag_key] = True
                        st.session_state[edit_text_key] = cmt_text
                        st.rerun()
                with del_col:
                    if st.button("🗑️ Delete", key=f"btn_del_{widget_key}_{cmt_id}",
                                 use_container_width=True):
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
                    save_col, cancel_col = st.columns([1, 1])
                    with save_col:
                        if st.button("💾 Save", key=f"btn_save_{widget_key}_{cmt_id}",
                                     use_container_width=True, type="primary"):
                            if update_movie_comment(cmt_id, new_text):
                                st.session_state[flash_key] = "Comment updated!"
                                st.session_state.pop(edit_flag_key, None)
                                st.session_state.pop(edit_text_key, None)
                            st.rerun()
                    with cancel_col:
                        if st.button("Cancel", key=f"btn_cancel_{widget_key}_{cmt_id}",
                                     use_container_width=True):
                            st.session_state.pop(edit_flag_key, None)
                            st.session_state.pop(edit_text_key, None)
                            st.rerun()

                st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)
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
        if st.button("➕ Add Comment", key=f"btn_add_{widget_key}",
                     use_container_width=True, type="primary"):
            if add_movie_comment(movie_title, new_comment):
                st.session_state[flash_key] = "Comment added!"
            else:
                st.session_state[flash_key] = ("warning", "Comment cannot be empty")
            st.rerun()


# -----------------------------
# Enhanced Database Functions
# -----------------------------
def init_db():
    """Initialize the enhanced SQLite database schema only"""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            
            # First, check if the table exists and what columns it has
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='enhanced_movies'")
            table_exists = c.fetchone()
            
            if table_exists:
                # Table exists, check columns
                c.execute("PRAGMA table_info(enhanced_movies)")
                columns = [column[1] for column in c.fetchall()]
                
                # Add missing columns if they don't exist
                if 'details_id' not in columns:
                    c.execute("ALTER TABLE enhanced_movies ADD COLUMN details_id TEXT")
                if 'details_rating' not in columns:
                    c.execute("ALTER TABLE enhanced_movies ADD COLUMN details_rating TEXT")
                if 'streaming_services' not in columns:
                    c.execute("ALTER TABLE enhanced_movies ADD COLUMN streaming_services TEXT")
                if 'ticket_available' not in columns:
                    c.execute("ALTER TABLE enhanced_movies ADD COLUMN ticket_available BOOLEAN DEFAULT FALSE")
                
                # If old IMDB columns exist, migrate data and remove them
                if 'imdb_id' in columns and 'details_id' in columns:
                    c.execute("UPDATE enhanced_movies SET details_id = imdb_id WHERE details_id IS NULL")
                if 'imdb_rating' in columns and 'details_rating' in columns:
                    c.execute("UPDATE enhanced_movies SET details_rating = imdb_rating WHERE details_rating IS NULL")
                    
            else:
                # Create new table with updated schema
                c.execute('''
                    CREATE TABLE IF NOT EXISTS enhanced_movies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        genre TEXT NOT NULL,
                        year INTEGER,
                        watched BOOLEAN DEFAULT FALSE,
                        rating INTEGER DEFAULT 0,
                        review TEXT DEFAULT '',
                        details_id TEXT,
                        poster_url TEXT,
                        plot TEXT,
                        director TEXT,
                        actors TEXT,
                        runtime TEXT,
                        details_rating TEXT,
                        streaming_services TEXT,
                        ticket_available BOOLEAN DEFAULT FALSE,
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            
            # Create indexes for better performance
            c.execute('CREATE INDEX IF NOT EXISTS idx_title ON enhanced_movies(title)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_genre ON enhanced_movies(genre)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_watched ON enhanced_movies(watched)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_year ON enhanced_movies(year)')
            c.execute(
                '''
                CREATE TABLE IF NOT EXISTS recommendation_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    movie_title TEXT NOT NULL,
                    movie_genre TEXT,
                    movie_year TEXT,
                    vote_type TEXT NOT NULL CHECK (vote_type IN ('like', 'dislike')),
                    source_query TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_recommendation_feedback_title ON recommendation_feedback(movie_title)'
            )
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_recommendation_feedback_created ON recommendation_feedback(created_at DESC)'
            )

            # Movie Comments table (AI Finder)
            c.execute(
                '''
                CREATE TABLE IF NOT EXISTS movie_comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    movie_title TEXT NOT NULL,
                    comment_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_movie_comments_title ON movie_comments(movie_title)'
            )
            c.execute(
                'CREATE INDEX IF NOT EXISTS idx_movie_comments_created ON movie_comments(created_at DESC)'
            )

            # Check if database needs initial setup marker
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='app_metadata'")
            if not c.fetchone():
                # First time setup
                c.execute('''
                    CREATE TABLE app_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                c.execute("INSERT INTO app_metadata (key, value) VALUES ('db_version', '1.0')")
                c.execute("INSERT INTO app_metadata (key, value) VALUES ('seeded', 'false')")
                conn.commit()
            
            conn.commit()
                    
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        st.error(f"Database error: {e}")

def check_and_seed_database():
    """Separate function to check and seed if needed"""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            c.execute("SELECT value FROM app_metadata WHERE key = 'seeded'")
            result = c.fetchone()
            
            if result and result[0] == 'false':
                c.execute("SELECT COUNT(*) FROM enhanced_movies")
                if c.fetchone()[0] == 0:
                    st.info("🌱 Setting up sample data...")
                    if seed.seed_database(CONFIG['database_url']):
                        c.execute("UPDATE app_metadata SET value = 'true' WHERE key = 'seeded'")
                        conn.commit()
                        st.success("✅ Setup complete!")
                        return True
            return False
    except Exception as e:
        logging.error(f"Seeding error: {e}")
        return False

def safe_db_operation(operation, *args):
    """Safe database operation with error handling"""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            result = operation(c, *args)
            conn.commit()
            return result
    except sqlite3.Error as e:
        logging.error(f"Database operation error: {e}")
        st.error(f"Database error: {e}")
        return None

def add_movie(title, genre, year, watched=False, rating=0, review="", details_data=None):
    """Add a movie to the database with enhanced information"""
    def operation(c, title, genre, year, watched, rating, review, details_data):
        streaming_finder = EnhancedStreamingServiceFinder()
        in_theaters = streaming_finder.is_movie_in_theaters(year)
        
        if details_data:
            c.execute('''
                INSERT INTO enhanced_movies 
                (title, genre, year, watched, rating, review, details_id, poster_url, plot, director, actors, runtime, details_rating, ticket_available)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                title, genre, year, watched, rating, review,
                details_data.get('imdbID'),
                details_data.get('Poster'),
                details_data.get('Plot'),
                details_data.get('Director'),
                details_data.get('Actors'),
                details_data.get('Runtime'),
                details_data.get('imdbRating'),
                in_theaters
            ))
        else:
            c.execute('''
                INSERT INTO enhanced_movies (title, genre, year, watched, rating, review, ticket_available)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (title, genre, year, watched, rating, review, in_theaters))
        return c.lastrowid
    return safe_db_operation(operation, title, genre, year, watched, rating, review, details_data)

def get_movies_safe():
    """Safe database connection with context manager"""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM enhanced_movies ORDER BY added_at DESC')
            return c.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Database error in get_movies: {e}")
        st.error(f"Database error: {e}")
        return []

@st.cache_data(ttl=60)
def get_movies_paginated(page=0, page_size=20, filters=None):
    """Efficient pagination with filters"""
    try:
        with sqlite3.connect(CONFIG['database_url']) as conn:
            c = conn.cursor()
            
            # Build query with filters
            query = "SELECT * FROM enhanced_movies WHERE 1=1"
            params = []
            
            if filters:
                if filters.get('search'):
                    query += " AND (title LIKE ? OR genre LIKE ?)"
                    search_term = f"%{filters['search']}%"
                    params.extend([search_term, search_term])
                if filters.get('genre') and filters['genre'] != 'All':
                    query += " AND genre = ?"
                    params.append(filters['genre'])
                if filters.get('watched') == 'Watched':
                    query += " AND watched = 1"
                elif filters.get('watched') == 'Unwatched':
                    query += " AND watched = 0"
            
            # Get total count for pagination
            count_query = query.replace("SELECT *", "SELECT COUNT(*)")
            c.execute(count_query, params)
            total_count = c.fetchone()[0]
            
            # Add pagination
            query += " ORDER BY added_at DESC LIMIT ? OFFSET ?"
            params.extend([page_size, page * page_size])
            
            c.execute(query, params)
            return c.fetchall(), total_count
    except sqlite3.Error as e:
        logging.error(f"Database error in pagination: {e}")
        return [], 0

def update_movie(movie_id, title, genre, year, rating=0, review=""):
    """Update a movie in the database"""
    def operation(c, movie_id, title, genre, year, rating, review):
        c.execute('''
            UPDATE enhanced_movies 
            SET title = ?, genre = ?, year = ?, rating = ?, review = ?, last_modified = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (title, genre, year, rating, review, movie_id))
        return c.rowcount
    return safe_db_operation(operation, movie_id, title, genre, year, rating, review)

def delete_movie(movie_id):
    """Delete a movie from the database"""
    def operation(c, movie_id):
        c.execute('DELETE FROM enhanced_movies WHERE id = ?', (movie_id,))
        return c.rowcount
    return safe_db_operation(operation, movie_id)

def toggle_watched(movie_id):
    """Toggle watched status of a movie"""
    def operation(c, movie_id):
        # Get current status
        c.execute('SELECT watched FROM enhanced_movies WHERE id = ?', (movie_id,))
        result = c.fetchone()
        if result:
            current_status = result[0]
            # Toggle status
            c.execute('UPDATE enhanced_movies SET watched = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?', 
                     (not current_status, movie_id))
            return c.rowcount
        return 0
    return safe_db_operation(operation, movie_id)

def add_movie_rating(movie_id, rating, review=""):
    """Add user rating and review for a movie"""
    def operation(c, movie_id, rating, review):
        c.execute('''
            UPDATE enhanced_movies 
            SET rating = ?, review = ?, last_modified = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (rating, review, movie_id))
        return c.rowcount
    return safe_db_operation(operation, movie_id, rating, review)

def get_stats():
    """Get enhanced movie statistics"""
    movies = get_movies_safe()
    if not movies:
        return {
            'total_movies': 0,
            'watched_count': 0,
            'completion_rate': 0,
            'unique_genres': 0,
            'average_rating': 0,
            'in_theaters_count': 0
        }
    
    total_movies = len(movies)
    watched_count = sum(1 for movie in movies if len(movie) > 4 and movie[4] == 1)
    completion_rate = (watched_count / total_movies * 100) if total_movies > 0 else 0
    unique_genres = len(set(movie[2] for movie in movies if len(movie) > 2))
    in_theaters_count = sum(1 for movie in movies if len(movie) > 15 and movie[15] == 1)
    
    # Calculate average rating (excluding unrated movies)
    ratings = [movie[5] for movie in movies if len(movie) > 5 and movie[5] > 0]
    average_rating = sum(ratings) / len(ratings) if ratings else 0
    
    return {
        'total_movies': total_movies,
        'watched_count': watched_count,
        'completion_rate': completion_rate,
        'unique_genres': unique_genres,
        'average_rating': average_rating,
        'in_theaters_count': in_theaters_count
    }

# -----------------------------
# Enhanced Movie Card Component with Streaming Integration
# -----------------------------
def create_movie_card(movie, key_suffix=""):
    """Create a clickable movie card with enhanced features"""
    if len(movie) < 6:
        return
    
    movie_id = movie[0]
    title = movie[1]
    genre = movie[2]
    year = movie[3]
    watched = movie[4] == 1
    rating = movie[5] if len(movie) > 5 else 0
    review = movie[6] if len(movie) > 6 else ""
    details_id = movie[7] if len(movie) > 7 else None
    poster_url = movie[8] if len(movie) > 8 else None
    plot = movie[9] if len(movie) > 9 else ""
    director = movie[10] if len(movie) > 10 else ""
    actors = movie[11] if len(movie) > 11 else ""
    runtime = movie[12] if len(movie) > 12 else ""
    details_rating = movie[13] if len(movie) > 13 else ""
    ticket_available = movie[15] if len(movie) > 15 else False
    added_date = movie[16] if len(movie) > 16 else ""
    
    # Initialize streaming finder
    streaming_finder = EnhancedStreamingServiceFinder()
    
    # Get app state
    app_state = st.session_state.app_state
    
    # Create card container
    with st.container():
        st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            status_icon = "✅" if watched else "⏳"
            st.markdown(f"### {status_icon} {title} ({year})")
            st.markdown(f"**Genre:** {genre}")
            
            # Display rating stars
            if rating > 0:
                stars = "★" * rating + "☆" * (5 - rating)
                st.markdown(f"**Your Rating:** <span class='star-rating'>{stars}</span>", unsafe_allow_html=True)
            
            if details_rating and details_rating != "N/A":
                st.markdown(f'<span class="rating-badge">IMDB: {details_rating}</span>', unsafe_allow_html=True)
            
            if added_date:
                st.markdown(f"**Added:** {added_date[:10]}")
            
            # Quick streaming info
            watch_options = streaming_finder.get_watch_options(title, year, genre)
            available_streaming = [s for s in watch_options['streaming'].values() if s['available']]
            if available_streaming:
                service_names = [s['name'] for s in available_streaming[:2]]
                st.markdown(f"**Available on:** {', '.join(service_names)}")
            
            if ticket_available:
                st.markdown("🎟️ **In Theaters Now**")
            
            # Quick Watch Options
            if st.button("🎯 Watch Now", key=f"watch_now_{movie_id}_{key_suffix}", use_container_width=True, type="primary"):
                app_state.show_watch_options.add(movie_id)
                st.rerun()
            
            # Details Button
            if st.button("🔍 Details", key=f"details_{movie_id}_{key_suffix}", use_container_width=True):
                app_state.show_details.add(movie_id)
                st.rerun()
        
        with col2:
            # Display poster if available
            if poster_url and poster_url != "N/A":
                st.image(poster_url, width="stretch")
            else:
                st.markdown("🎭 *No poster*")
        
        with col3:
            st.markdown("### ")
            # Action buttons
            button_text = "⏳ Unwatch" if watched else "✅ Watch"
            button_color = "secondary" if watched else "primary"
            if st.button(button_text, key=f"watch_{movie_id}_{key_suffix}", 
                        use_container_width=True, type=button_color):
                toggle_watched(movie_id)
                st.rerun()
            
            if st.button("⭐ Rate", key=f"rate_{movie_id}_{key_suffix}", 
                        use_container_width=True, type="secondary"):
                app_state.show_rating.add(movie_id)
                st.rerun()
            
            if st.button("🗑️ Delete", key=f"delete_{movie_id}_{key_suffix}", 
                        use_container_width=True, type="secondary"):
                delete_movie(movie_id)
                st.success(f"Deleted {title} from collection!")
                st.rerun()
        
        # Watch Options Section
        if movie_id in app_state.show_watch_options:
            display_watch_options_section(title, year, genre, movie_id, key_suffix)
        
        # Rating Section
        if movie_id in app_state.show_rating:
            with st.container():
                st.markdown("---")
                st.markdown("#### ⭐ Rate this Movie")
                col_rate1, col_rate2 = st.columns([2, 1])
                with col_rate1:
                    new_rating = st.slider("Your Rating", 1, 5, rating, key=f"slider_{movie_id}")
                    review_text = st.text_area("Your Review (optional)", review, key=f"review_{movie_id}")
                with col_rate2:
                    st.markdown("### ")
                    if st.button("💾 Save Rating", key=f"save_rating_{movie_id}", use_container_width=True, type="primary"):
                        add_movie_rating(movie_id, new_rating, review_text)
                        app_state.show_rating.discard(movie_id)
                        st.success("Rating saved!")
                        st.rerun()
                    if st.button("❌ Cancel", key=f"cancel_rating_{movie_id}", use_container_width=True):
                        app_state.show_rating.discard(movie_id)
                        st.rerun()
        
        # Details Section
        if movie_id in app_state.show_details:
            display_details_section(title, year, movie_id, key_suffix)
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_watch_options_section(title, year, genre, movie_id, key_suffix):
    """Display streaming and ticketing options for a movie"""
    streaming_finder = EnhancedStreamingServiceFinder()
    
    with st.spinner("🔍 Finding where to watch..."):
        watch_options = streaming_finder.get_watch_options(title, year, genre)
    
    with st.container():
        st.markdown('<div class="watch-now-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 Where to Watch")
        
        # Streaming Services
        st.markdown("#### 📺 Streaming Services")
        streaming_cols = st.columns(3)
        col_idx = 0
        
        available_services = [s for s in watch_options['streaming'].values() if s['available']]
        unavailable_services = [s for s in watch_options['streaming'].values() if not s['available']]
        
        if available_services:
            for service in available_services:
                with streaming_cols[col_idx % 3]:
                    badge_class = "available-service"
                    if service['name'] == 'Crunchyroll':
                        badge_class = "anime-service"
                    badge_html = f"""
                    <a href="{service['url']}" target="_blank" style="text-decoration: none;">
                        <div class="streaming-badge {badge_class}" style="border-color: {service['color']};">
                            {service['icon']} {service['name']} ✅
                        </div>
                    </a>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)
                col_idx += 1
        else:
            st.info("No streaming services found for this movie.")
        
        # Anime Services
        if 'anime' in watch_options:
            st.markdown("#### 🍥 Anime Streaming")
            anime_cols = st.columns(2)
            for service in watch_options['anime'].values():
                with anime_cols[0]:
                    badge_html = f"""
                    <a href="{service['url']}" target="_blank" style="text-decoration: none;">
                        <div class="streaming-badge anime-service">
                            {service['icon']} {service['name']} ✅
                        </div>
                    </a>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)
        
        # Ticketing Services (if movie is in theaters)
        if watch_options['in_theaters'] and watch_options['ticketing']:
            st.markdown("#### 🎟️ Buy Tickets")
            ticket_cols = st.columns(2)
            col_idx = 0
            
            for service in watch_options['ticketing'].values():
                with ticket_cols[col_idx % 2]:
                    badge_html = f"""
                    <a href="{service['url']}" target="_blank" style="text-decoration: none;">
                        <div class="streaming-badge ticket-service">
                            {service['icon']} {service['name']}
                        </div>
                    </a>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)
                col_idx += 1
        
        # Alternative options
        st.markdown("#### 🔍 Other Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            justwatch_url = f"https://www.justwatch.com/us/search?q={quote(title)}"
            st.markdown(f'<a href="{justwatch_url}" target="_blank"><div class="streaming-badge available-service">🔍 JustWatch</div></a>', unsafe_allow_html=True)
        
        with col2:
            google_url = f"https://www.google.com/search?q={quote(title)}+streaming"
            st.markdown(f'<a href="{google_url}" target="_blank"><div class="streaming-badge available-service">🌐 Google Search</div></a>', unsafe_allow_html=True)
        
        with col3:
            youtube_url = f"https://www.youtube.com/results?search_query={quote(title + ' trailer')}"
            st.markdown(f'<a href="{youtube_url}" target="_blank"><div class="streaming-badge available-service">📺 Trailer</div></a>', unsafe_allow_html=True)
        
        # Close button
        app_state = st.session_state.app_state
        if st.button("Close Watch Options", key=f"close_watch_{movie_id}_{key_suffix}", use_container_width=True, type="secondary"):
            app_state.show_watch_options.discard(movie_id)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_details_section(title, year, movie_id, key_suffix):
    """Display OMDB details for a movie"""
    omdb = RateLimitedOMDbAPI()
    with st.spinner("🎬 Fetching movie details..."):
        movie_data = omdb.get_movie_with_streaming_info(title, year)
    
    if movie_data:
        with st.container():
            st.markdown('<div class="movie-detail-card">', unsafe_allow_html=True)
            st.markdown("### 🎬 Movie Details")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display poster if available
                if movie_data.get("Poster") and movie_data["Poster"] != "N/A":
                    st.image(movie_data["Poster"], width="stretch")
                else:
                    st.markdown("🎭 *No poster available*")
                
                # Quick Watch Button
                if st.button("🎯 Watch Now", key=f"details_watch_{movie_id}", use_container_width=True, type="primary"):
                    st.session_state.app_state.show_watch_options.add(movie_id)
                    st.rerun()
            
            with col2:
                # Movie information
                st.markdown(f"**Title:** {movie_data.get('Title', 'N/A')}")
                st.markdown(f"**Year:** {movie_data.get('Year', 'N/A')}")
                st.markdown(f"**Rated:** {movie_data.get('Rated', 'N/A')}")
                st.markdown(f"**Runtime:** {movie_data.get('Runtime', 'N/A')}")
                st.markdown(f"**Genre:** {movie_data.get('Genre', 'N/A')}")
                st.markdown(f"**Director:** {movie_data.get('Director', 'N/A')}")
                st.markdown(f"**Actors:** {movie_data.get('Actors', 'N/A')}")
                
                # Ratings with enhanced display
                ratings = movie_data.get('Ratings', [])
                if ratings:
                    st.markdown("**Ratings:**")
                    for rating in ratings:
                        source = rating.get('Source', '')
                        value = rating.get('Value', '')
                        if 'Internet Movie Database' in source:
                            st.markdown(f'<span class="rating-badge">🎬 IMDB: {value}</span>', unsafe_allow_html=True)
                        elif 'Rotten Tomatoes' in source:
                            st.markdown(f'<span style="color: #FF6B6B">🍅 Rotten Tomatoes: {value}</span>', unsafe_allow_html=True)
                        elif 'Metacritic' in source:
                            st.markdown(f'<span style="color: #4D96FF">💎 Metacritic: {value}</span>', unsafe_allow_html=True)
                
                # IMDB Rating (if available separately)
                imdb_rating = movie_data.get('imdbRating', 'N/A')
                if imdb_rating != 'N/A':
                    st.markdown(f'<span class="rating-badge">⭐ IMDB Rating: {imdb_rating}/10</span>', unsafe_allow_html=True)
                
                st.markdown(f"**Plot:** {movie_data.get('Plot', 'N/A')}")
            
            # Streaming information if available
            if 'streaming_info' in movie_data:
                streaming_info = movie_data['streaming_info']
                available_streaming = [s for s in streaming_info['streaming'].values() if s['available']]
                
                if available_streaming:
                    st.markdown("#### 📺 Available On")
                    stream_cols = st.columns(min(4, len(available_streaming)))
                    for idx, service in enumerate(available_streaming):
                        with stream_cols[idx % len(stream_cols)]:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; border-radius: 10px; background: {service['color']}20; border: 1px solid {service['color']}50;">
                                <div style="font-size: 1.5em;">{service['icon']}</div>
                                <div><strong>{service['name']}</strong></div>
                                <a href="{service['url']}" target="_blank" style="color: {service['color']}; text-decoration: none;">Watch →</a>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Close button
            app_state = st.session_state.app_state
            if st.button("Close Details", key=f"close_{movie_id}_{key_suffix}", use_container_width=True, type="secondary"):
                app_state.show_details.discard(movie_id)
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("❌ Could not fetch movie details. Please try again.")

# -----------------------------
# Enhanced AI Finder Page with OMDB Integration - SIMPLIFIED
# -----------------------------
def show_enhanced_ai_finder_page():
    """Enhanced AI Finder page with cinematic styling, OMDB search + Gemini chatbot."""
    feedback_profile, feedback_states = get_recommendation_feedback_context()
    feedback_summary_text = build_feedback_summary_text(feedback_profile)
    default_filter_values = get_ai_finder_filter_defaults()

    st.markdown("""
    <div class="hero-container">
        <h2 style="text-align: center; font-size: 2.5rem; margin-bottom: 1rem;">🔍 ENHANCED AI FINDER</h2>
        <p style="text-align: center; font-size: 1.3rem; color: #FFD93D; margin: 0;">
            Discover Movies & Anime Across All Platforms
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.get("feedback_flash"):
        st.success(st.session_state.pop("feedback_flash"))
    if feedback_summary_text:
        st.caption(feedback_summary_text)

    with st.expander("Recommendation Filters", expanded=True):
        filter_cols = st.columns([2, 1, 1, 1])
        with filter_cols[0]:
            year_range = st.slider(
                "Year range",
                min_value=default_filter_values["year_range"][0],
                max_value=default_filter_values["year_range"][1],
                key="ai_filter_year_range",
            )
        with filter_cols[1]:
            min_imdb_rating = st.slider(
                "Minimum IMDb",
                min_value=0.0,
                max_value=10.0,
                step=0.5,
                key="ai_filter_min_imdb",
            )
        with filter_cols[2]:
            content_type = st.selectbox(
                "Content type",
                ["All", "Movie", "Anime"],
                key="ai_filter_content_type",
            )
        with filter_cols[3]:
            st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
            st.button(
                "Reset Filters",
                key="reset_ai_filters",
                use_container_width=True,
                on_click=reset_ai_finder_filters,
            )

        recommendation_filters = {
            "year_range": year_range,
            "min_imdb_rating": float(min_imdb_rating),
            "content_type": content_type,
        }
        st.session_state.ai_finder_filters = recommendation_filters
        st.caption(build_ai_filter_summary(recommendation_filters))

    # Two columns: left = search (old behaviour), right = Gemini chatbot
    col1, col2 = st.columns([2, 1])

    # -----------------------------
    # LEFT – your existing simple search + popular/theater tabs
    # -----------------------------
    with col1:
        st.markdown("### 🤖 ENHANCED MOVIE & ANIME SEARCH")

        search_query = st.text_input(
            "🎬 Search movies or anime.",
            placeholder="Enter movie title, anime, or genre.",
            key="enhanced_ai_search",
        )

        if search_query:
            with st.spinner("🔍 Searching across all platforms."):
                # OMDB search
                omdb = RateLimitedOMDbAPI()
                movie_results = omdb.search_movies(search_query, "movie", None)

                # Anime search
                anime_integration = AnimeIntegration()
                anime_results = anime_integration.search_anime(search_query)

                if movie_results or anime_results:
                    st.markdown(
                        f"### 🎯 Found {len(movie_results) + len(anime_results)} Results"
                    )

                    if movie_results:
                        st.markdown("#### 🎬 Movies & Series")
                        for i, movie in enumerate(movie_results):
                            display_enhanced_search_result(movie, "movie", i)

                    if anime_results:
                        st.markdown("#### 🍥 Anime")
                        for i, anime in enumerate(anime_results):
                            display_enhanced_search_result(anime, "anime", i)
                else:
                    st.warning(
                        "No results found. Try a different search term or check our recommendations below."
                    )
                    # If you have this helper it will still work;
                    # if not, this will just be skipped.
                    try:
                        show_search_recommendations(search_query, recommendation_filters=recommendation_filters)
                    except NameError:
                        pass
        else:
            # Old behaviour: show tabs with Popular / In Theaters etc.
            show_popular_content()

    # -----------------------------
    # RIGHT – Gemini movie chatbot
    # -----------------------------
    with col2:
        st.markdown("### 💬 Movie Help Chatbot (Gemini)")

        # Lazy init so we don't touch your global init_session_state
        if "gemini_chat" not in st.session_state:
            st.session_state.gemini_chat = GeminiMovieChat()
        if "ai_history" not in st.session_state:
            st.session_state.ai_history = []
        if "ai_edit_turn_index" not in st.session_state:
            st.session_state.ai_edit_turn_index = None
        if "ai_edit_prompt" not in st.session_state:
            st.session_state.ai_edit_prompt = ""
        chat = st.session_state.gemini_chat
        sync_ai_finder_chat_history(chat)

        if st.session_state.get("ai_chat_flash"):
            flash_message = st.session_state.pop("ai_chat_flash")
            flash_level = st.session_state.pop("ai_chat_flash_level", "success")
            if flash_level == "error":
                st.error(flash_message)
            else:
                st.success(flash_message)

        st.markdown("#### Quick Actions")
        st.caption("Send a ready-made prompt. Similar and Anime use the left search text when available.")

        quick_action_prompt = None
        quick_cols = st.columns(3)
        with quick_cols[0]:
            if st.button("Recommend by Mood", key="quick_action_mood", use_container_width=True):
                quick_action_prompt = build_quick_action_prompt("mood", search_query)
        with quick_cols[1]:
            if st.button("Similar to This Movie", key="quick_action_similar", use_container_width=True):
                quick_action_prompt = build_quick_action_prompt("similar", search_query)
        with quick_cols[2]:
            if st.button("Anime-Only", key="quick_action_anime", use_container_width=True):
                quick_action_prompt = build_quick_action_prompt("anime", search_query)

        # Show chat history
        for turn in get_ai_finder_history_turns():
            user_index = turn["user_index"]
            user_message = turn["user_message"]
            assistant_message = turn["assistant_message"]

            with st.chat_message("user"):
                st.markdown(user_message["content"])
                action_cols = st.columns([1, 1, 2])
                with action_cols[0]:
                    if st.button("Edit", key=f"edit_ai_turn_{user_index}", use_container_width=True):
                        st.session_state.ai_edit_turn_index = user_index
                        st.session_state.ai_edit_prompt = user_message["content"]
                        st.rerun()
                with action_cols[1]:
                    if st.button("Delete", key=f"delete_ai_turn_{user_index}", use_container_width=True):
                        if delete_ai_finder_history_turn(chat, user_index):
                            st.session_state.ai_chat_flash = "Deleted that prompt and reply."
                            st.session_state.ai_chat_flash_level = "success"
                        else:
                            st.session_state.ai_chat_flash = "Could not delete that chat turn. Please try again."
                            st.session_state.ai_chat_flash_level = "error"
                        st.rerun()

                if st.session_state.ai_edit_turn_index == user_index:
                    edited_prompt = st.text_area(
                        "Edit your prompt",
                        value=st.session_state.ai_edit_prompt or user_message["content"],
                        key=f"edit_ai_prompt_{user_index}",
                        height=120,
                    )
                    save_cols = st.columns(2)
                    with save_cols[0]:
                        if st.button("Save Changes", key=f"save_ai_turn_{user_index}", use_container_width=True, type="primary"):
                            turn_filters = (assistant_message or {}).get("filters") or recommendation_filters
                            updated_message, feedback_profile = update_ai_finder_history_turn(
                                chat,
                                user_index,
                                edited_prompt,
                                feedback_profile,
                                recommendation_filters=turn_filters,
                            )
                            if updated_message:
                                feedback_states = get_recommendation_feedback_context()[1]
                                st.session_state.ai_chat_flash = "Updated the prompt and regenerated the reply."
                                st.session_state.ai_chat_flash_level = "success"
                            else:
                                st.session_state.ai_chat_flash = "Prompt cannot be empty."
                                st.session_state.ai_chat_flash_level = "error"
                            st.rerun()
                    with save_cols[1]:
                        if st.button("Cancel", key=f"cancel_ai_turn_{user_index}", use_container_width=True):
                            clear_ai_finder_edit_state()
                            st.rerun()

            if assistant_message:
                with st.chat_message("assistant"):
                    st.markdown(assistant_message["content"])
                    if assistant_message.get("recommendations"):
                        render_explainable_recommendations(
                            assistant_message["recommendations"],
                            key_prefix=f"chat_history_{user_index}",
                            feedback_states=feedback_states,
                            feedback_query=assistant_message.get("source_query", ""),
                        )

        # User input (works like before)
        user_msg = st.chat_input(
            "Tell me your mood, a plot, or actor names...\nExample: 'Sad mood, recommend 3 sci-fi movies' or 'Movie like Interstellar but with more romance'."
        )

        submitted_prompt = user_msg or quick_action_prompt
        if submitted_prompt:
            assistant_message, feedback_profile = run_ai_finder_chat_turn(
                chat,
                submitted_prompt,
                feedback_profile,
                recommendation_filters=recommendation_filters,
            )
            feedback_states = get_recommendation_feedback_context()[1]

            with st.chat_message("user"):
                st.markdown(submitted_prompt)
            with st.chat_message("assistant"):
                st.markdown(assistant_message["content"])

                if assistant_message["recommendations"]:
                    st.markdown("**Suggested movies (max 3):**")
                    render_explainable_recommendations(
                        assistant_message["recommendations"],
                        key_prefix="chat_live",
                        feedback_states=feedback_states,
                        feedback_query=submitted_prompt,
                    )
                else:
                    st.markdown(
                        "_No specific movie titles could be found for this query._"
                    )

        st.markdown("---")
        st.markdown("##### ℹ️ What can I ask?")
        st.caption(
            """
- *"I feel lonely, suggest 3 movies."*
- *"Give me 3 comedy movies with Jim Carrey."*
- *"Anime where the main character travels through time."*
- *"Movie similar to Inception but easier to understand."*
"""
        )


def display_enhanced_search_result(item, item_type, index):
    """Display enhanced search results with OMDB details and streaming options"""
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns([3, 1, 1])
        
        with col_a:
            title = item.get('Title') if item_type == 'movie' else item.get('title')
            year = item.get('Year') if item_type == 'movie' else item.get('year')
            genre = item.get('Genre') if item_type == 'movie' else item.get('genre')
            
            st.markdown(f"#### {title} ({year})")
            st.markdown(f"**Type:** {item_type.title()} | **Genre:** {genre}")
            
            # Get detailed movie information from OMDB
            if item_type == 'movie':
                omdb = RateLimitedOMDbAPI()
                with st.spinner(f"Fetching details for {title}..."):
                    movie_details = omdb.get_movie_details(title, int(year) if year and year.isdigit() else None)
                
                if movie_details:
                    # Display ratings
                    ratings = movie_details.get('Ratings', [])
                    imdb_rating = movie_details.get('imdbRating', 'N/A')
                    
                    if imdb_rating != 'N/A':
                        st.markdown(f'<span class="rating-badge">⭐ IMDB: {imdb_rating}/10</span>', unsafe_allow_html=True)
                    
                    # Display other ratings
                    for rating in ratings:
                        source = rating.get('Source', '')
                        value = rating.get('Value', '')
                        if 'Rotten Tomatoes' in source:
                            st.markdown(f'<span style="color: #FF6B6B">🍅 {value}</span>', unsafe_allow_html=True)
                    
                    # Display other details
                    if movie_details.get('Director') and movie_details['Director'] != 'N/A':
                        st.markdown(f"**Director:** {movie_details['Director']}")
                    if movie_details.get('Runtime') and movie_details['Runtime'] != 'N/A':
                        st.markdown(f"**Runtime:** {movie_details['Runtime']}")
                    if movie_details.get('Plot') and movie_details['Plot'] != 'N/A':
                        plot_preview = movie_details['Plot'][:100] + "..." if len(movie_details['Plot']) > 100 else movie_details['Plot']
                        st.markdown(f"**Plot:** {plot_preview}")
            
            # Show streaming availability
            streaming_finder = EnhancedStreamingServiceFinder()
            watch_options = streaming_finder.get_watch_options(title, int(year) if year and year.isdigit() else 2023, genre)
            
            # Display available services
            available_services = []
            for service_type, services in watch_options.items():
                if service_type == 'streaming':
                    available_services.extend([s for s in services.values() if s['available']])
                elif service_type == 'anime':
                    available_services.extend(services.values())
            
            if available_services:
                service_names = [f"{s['icon']} {s['name']}" for s in available_services[:2]]
                st.markdown(f"**Available on:** {', '.join(service_names)}")
        
        with col_b:
            # Display poster if available
            poster = item.get('Poster')
            if poster and poster != "N/A":
                st.image(poster, width=80)
            else:
                # Try to get poster from OMDB details
                if item_type == 'movie':
                    omdb = RateLimitedOMDbAPI()
                    movie_details = omdb.get_movie_details(title, int(year) if year and year.isdigit() else None)
                    if movie_details and movie_details.get('Poster') and movie_details['Poster'] != 'N/A':
                        st.image(movie_details['Poster'], width=80)
                    else:
                        st.markdown("🎭 *No poster*")
                else:
                    st.markdown("🎭 *No poster*")
        
        with col_c:
            if st.button("➕ Add", key=f"add_{item_type}_{index}", use_container_width=True, type="primary"):
                # Extract year properly
                year_str = year
                if year_str and isinstance(year_str, str):
                    year_val = int(''.join(filter(str.isdigit, year_str))[:4]) if any(c.isdigit() for c in year_str) else 2023
                else:
                    year_val = year if year else 2023
                
                # Get detailed movie data for storage
                omdb = RateLimitedOMDbAPI()
                details_data = None
                if item_type == 'movie':
                    details_data = omdb.get_movie_details(title, year_val)
                
                # Add to collection
                add_movie(
                    title=title,
                    genre=genre or "Unknown",
                    year=year_val,
                    watched=False,
                    details_data=details_data
                )
                st.success(f"Added {title} to collection!")
                st.rerun()
            
            if st.button("🎯 Watch", key=f"watch_{item_type}_{index}", use_container_width=True, type="secondary"):
                st.session_state[f"show_watch_{title}"] = True
            
            if st.button("🔍 Details", key=f"details_{item_type}_{index}", use_container_width=True, type="secondary"):
                st.session_state[f"show_full_details_{title}"] = True
        
        # Show watch options if triggered
        if st.session_state.get(f"show_watch_{title}"):
            year_val = int(year) if year and year.isdigit() else 2023
            display_watch_options_section(title, year_val, genre or "Unknown", f"search_{index}", f"search_{index}")
        
        # Show full details if triggered
        if st.session_state.get(f"show_full_details_{title}"):
            year_val = int(year) if year and year.isdigit() else 2023
            display_full_movie_details(title, year_val, f"search_{index}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_full_movie_details(title, year, key_suffix):
    """Display full movie details from OMDB"""
    omdb = RateLimitedOMDbAPI()
    with st.spinner("🎬 Fetching complete movie details..."):
        movie_data = omdb.get_movie_details(title, year)
    
    if movie_data:
        with st.container():
            st.markdown('<div class="movie-detail-card">', unsafe_allow_html=True)
            st.markdown("### 🎬 Complete Movie Details")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display poster if available
                if movie_data.get("Poster") and movie_data["Poster"] != "N/A":
                    st.image(movie_data["Poster"], width="stretch")
                else:
                    st.markdown("🎭 *No poster available*")
            
            with col2:
                # Comprehensive movie information
                st.markdown(f"**Title:** {movie_data.get('Title', 'N/A')}")
                st.markdown(f"**Year:** {movie_data.get('Year', 'N/A')}")
                st.markdown(f"**Rated:** {movie_data.get('Rated', 'N/A')}")
                st.markdown(f"**Released:** {movie_data.get('Released', 'N/A')}")
                st.markdown(f"**Runtime:** {movie_data.get('Runtime', 'N/A')}")
                st.markdown(f"**Genre:** {movie_data.get('Genre', 'N/A')}")
                st.markdown(f"**Director:** {movie_data.get('Director', 'N/A')}")
                st.markdown(f"**Writer:** {movie_data.get('Writer', 'N/A')}")
                st.markdown(f"**Actors:** {movie_data.get('Actors', 'N/A')}")
                st.markdown(f"**Language:** {movie_data.get('Language', 'N/A')}")
                st.markdown(f"**Country:** {movie_data.get('Country', 'N/A')}")
                st.markdown(f"**Awards:** {movie_data.get('Awards', 'N/A')}")
                
                # Enhanced ratings display
                ratings = movie_data.get('Ratings', [])
                if ratings:
                    st.markdown("**Ratings:**")
                    for rating in ratings:
                        source = rating.get('Source', '')
                        value = rating.get('Value', '')
                        if 'Internet Movie Database' in source:
                            st.markdown(f'<span class="rating-badge">🎬 IMDB: {value}</span>', unsafe_allow_html=True)
                        elif 'Rotten Tomatoes' in source:
                            st.markdown(f'<span style="color: #FF6B6B; font-weight: bold;">🍅 Rotten Tomatoes: {value}</span>', unsafe_allow_html=True)
                        elif 'Metacritic' in source:
                            st.markdown(f'<span style="color: #4D96FF; font-weight: bold;">💎 Metacritic: {value}</span>', unsafe_allow_html=True)
                
                # IMDB specific ratings
                imdb_rating = movie_data.get('imdbRating', 'N/A')
                imdb_votes = movie_data.get('imdbVotes', 'N/A')
                if imdb_rating != 'N/A':
                    st.markdown(f'<span class="rating-badge">⭐ IMDB Rating: {imdb_rating}/10 ({imdb_votes} votes)</span>', unsafe_allow_html=True)
                
                # Box office information
                if movie_data.get('BoxOffice') and movie_data['BoxOffice'] != 'N/A':
                    st.markdown(f"**Box Office:** {movie_data['BoxOffice']}")
                
                st.markdown(f"**Plot:** {movie_data.get('Plot', 'N/A')}")
            
            # Close button
            if st.button("Close Details", key=f"close_full_{key_suffix}", use_container_width=True, type="secondary"):
                st.session_state[f"show_full_details_{title}"] = False
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

def show_popular_content():
    """Show popular streaming and theater content"""
    st.markdown("### 🔥 POPULAR RIGHT NOW")
    
    # Popular on Streaming
    st.markdown("#### 📺 POPULAR ON STREAMING")
    streaming_movies = [m for m in MOVIE_DATABASE if m.get('streaming_service') and not m.get('in_theaters', False)]
    
    cols = st.columns(3)
    for idx, movie in enumerate(streaming_movies[:6]):
        with cols[idx % 3]:
            st.markdown(f"**{movie['title']}**")
            st.markdown(f"*{movie['streaming_service']}*")
            if st.button("Add", key=f"stream_{idx}", use_container_width=True, type="primary"):
                add_movie(movie['title'], movie['genre'], movie['year'], False)
                st.success(f"Added {movie['title']}!")
    
    # In Theaters
    st.markdown("#### 🎟️ IN THEATERS NOW")
    theater_movies = [m for m in MOVIE_DATABASE if m.get('in_theaters', False)]
    
    theater_cols = st.columns(2)
    for idx, movie in enumerate(theater_movies[:4]):
        with theater_cols[idx % 2]:
            st.markdown(f"**{movie['title']}** ({movie['year']})")
            st.markdown(f"*{movie['genre']}*")
            if st.button("Get Tickets", key=f"theater_btn_{idx}", use_container_width=True, type="primary"):
                streaming_finder = EnhancedStreamingServiceFinder()
                watch_options = streaming_finder.get_watch_options(movie['title'], movie['year'], movie['genre'])
                st.session_state[f"show_tickets_{movie['title']}"] = True

def show_search_recommendations(query, recommendation_filters=None):
    """Show recommendations when no results found"""
    st.markdown("### 💡 TRY THESE POPULAR TITLES")
    
    # Get recommendations based on query
    recommendations = []
    query_lower = query.lower()
    
    # Genre-based recommendations
    genre_keywords = {
        'action': ['action', 'adventure', 'thriller', 'fight'],
        'comedy': ['comedy', 'funny', 'humor'],
        'drama': ['drama', 'emotional', 'serious'],
        'anime': ['anime', 'manga', 'japanese'],
        'horror': ['horror', 'scary', 'terror']
    }
    
    for genre, keywords in genre_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            genre_movies = [m for m in MOVIE_DATABASE if m['genre'].lower() == genre]
            recommendations.extend(genre_movies[:2])
    
    # If no genre match, show general popular movies
    if not recommendations:
        recommendations = [m for m in MOVIE_DATABASE if m.get('streaming_service')][:6]

    deduped_recommendations = []
    seen_titles = set()
    for movie in recommendations:
        title_key = movie['title'].strip().lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        deduped_recommendations.append(movie)

    explained_recommendations, feedback_profile = build_feedback_aware_recommendations(
        query,
        deduped_recommendations[:6],
        limit=6,
        filters=recommendation_filters,
    )
    feedback_states = feedback_profile.get("latest_votes", {})

    if not explained_recommendations:
        st.info("No fallback recommendations matched your current AI Finder filters.")
        return

    rec_cols = st.columns(3)
    for idx, movie in enumerate(explained_recommendations):
        with rec_cols[idx % 3]:
            render_explainable_recommendations(
                [movie],
                key_prefix=f"rec_{idx}",
                show_add_button=True,
                feedback_states=feedback_states,
                feedback_query=query,
            )

# -----------------------------
# Enhanced AI Chat System
# -----------------------------
class AdvancedAIChat:
    def __init__(self):
        self.conversation_history = []
        self.omdb = RateLimitedOMDbAPI()
        self.user_preferences = {
            "favorite_genres": [],
            "watch_habits": {},
            "recent_interests": []
        }
        self.streaming_finder = EnhancedStreamingServiceFinder()
        
    def generate_ai_response(self, user_input, movie_data):
        """Generate intelligent AI response with streaming integration"""
        user_input_lower = user_input.lower()
        
        # Update user preferences based on conversation
        self._update_user_preferences(user_input, movie_data)
        
        # Enhanced response system with streaming integration
        if any(word in user_input_lower for word in ['watch', 'stream', 'where to watch', 'netflix', 'amazon', 'hulu', 'disney', 'youtube']):
            return self._get_streaming_response(user_input, movie_data)
        
        elif any(word in user_input_lower for word in ['ticket', 'theater', 'cinema', 'buy ticket']):
            return self._get_ticketing_response(user_input, movie_data)
        
        elif any(word in user_input_lower for word in ['anime', 'crunchyroll']):
            return self._get_anime_response(user_input, movie_data)
        
        elif any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return self._get_greeting_response(movie_data)
        
        elif any(word in user_input_lower for word in ['recommend', 'suggest', 'what should i watch', 'what to watch']):
            return self._get_recommendation_response(user_input, movie_data)
        
        elif any(word in user_input_lower for word in ['action', 'comedy', 'drama', 'sci-fi', 'romance', 'horror', 'thriller']):
            return self._get_genre_response(user_input_lower, movie_data)
        
        elif any(word in user_input_lower for word in ['details', 'info', 'about movie', 'tell me about']):
            return self._get_movie_details_response(user_input)
        
        elif any(word in user_input_lower for word in ['search', 'find movie', 'look for']):
            return self._get_movie_search_response(user_input)
        
        elif any(word in user_input_lower for word in ['analyze', 'stats', 'statistics', 'my collection', 'how many']):
            return self._get_analysis_response(movie_data)
        
        elif any(word in user_input_lower for word in ['watched', 'unwatched', 'watchlist']):
            return self._get_watch_status_response(movie_data)
        
        elif any(word in user_input_lower for word in ['help', 'what can you do', 'features']):
            return self._get_help_response()
        
        elif any(word in user_input_lower for word in ['best', 'top', 'greatest']):
            return self._get_best_movies_response(user_input_lower)
        
        else:
            return self._get_intelligent_fallback(user_input, movie_data)
    
    def _get_streaming_response(self, user_input, movie_data):
        """Handle streaming-related queries"""
        words = user_input.split()
        movie_keywords = [word for word in words if word.lower() not in 
                         ['watch', 'stream', 'where', 'to', 'on', 'netflix', 'amazon', 'prime', 'disney', 'hulu', 'hbo', 'youtube', 'crunchyroll']]
        
        if movie_keywords:
            movie_title = " ".join(movie_keywords[:4])
            
            # Check if movie is in collection
            collection_movie = None
            if movie_data:
                for movie in movie_data:
                    if movie_title.lower() in movie[1].lower():
                        collection_movie = movie
                        break
            
            if collection_movie:
                title = collection_movie[1]
                year = collection_movie[3]
                genre = collection_movie[2]
                
                watch_options = self.streaming_finder.get_watch_options(title, year, genre)
                available_streaming = [s for s in watch_options['streaming'].values() if s['available']]
                
                response = f"🎬 **{title}** ({year})\n\n"
                
                if available_streaming:
                    response += "**Available on:**\n"
                    for service in available_streaming:
                        response += f"• {service['icon']} **{service['name']}** - [Watch Now]({service['url']})\n"
                else:
                    response += "**Streaming:** Not currently available on major platforms\n"
                
                if watch_options['in_theaters']:
                    response += "\n**🎟️ In Theaters Now!**\n"
                    response += "Get tickets from:\n"
                    for service in watch_options['ticketing'].values():
                        response += f"• {service['icon']} **{service['name']}** - [Buy Tickets]({service['url']})\n"
                
                # Check for anime
                if 'anime' in watch_options:
                    response += "\n**🍥 Anime Streaming:**\n"
                    for service in watch_options['anime'].values():
                        response += f"• {service['icon']} **{service['name']}** - [Watch Anime]({service['url']})\n"
                
                response += f"\n💡 *Click the 'Watch Now' button on the movie card for more options!*"
                return response
            else:
                return f"🎬 I couldn't find '{movie_title}' in your collection. Try searching for it in the AI Finder or add it to your collection first!"
        
        return "🎬 Tell me which movie you'd like to watch! For example: 'Where can I watch Inception?' or 'Is The Dark Knight on Netflix?'"

    def _get_ticketing_response(self, user_input, movie_data):
        """Handle ticketing-related queries"""
        current_year = datetime.now().year
        in_theaters_movies = [movie for movie in movie_data if len(movie) > 3 and movie[3] >= current_year - 1]
        
        if not in_theaters_movies:
            return "🎟️ No recent movies found in your collection that might be in theaters. Recent releases from 2023-2024 are most likely to be in theaters!"
        
        response = "🎟️ **Movies That Might Be In Theaters**\n\n"
        response += "These recent movies from your collection might be in theaters:\n\n"
        
        for movie in in_theaters_movies[:5]:  # Show top 5
            title = movie[1]
            year = movie[3]
            response += f"• **{title}** ({year})\n"
            response += f"  [Get Tickets](https://www.fandango.com/search?q={quote(title)})\n\n"
        
        response += "💡 *Click 'Watch Now' on any movie card to check all ticketing options!*"
        return response

    def _get_anime_response(self, user_input, movie_data):
        """Handle anime-related queries"""
        anime_integration = AnimeIntegration()
        
        words = user_input.split()
        anime_keywords = [word for word in words if word.lower() not in ['anime', 'watch', 'find', 'search']]
        
        if anime_keywords:
            anime_query = " ".join(anime_keywords)
            anime_results = anime_integration.search_anime(anime_query)
            
            if anime_results:
                response = f"🍥 **Anime Results for '{anime_query}'**\n\n"
                for anime in anime_results[:3]:
                    response += f"• **{anime['title']}** ({anime['year']}) - {anime['genre']}\n"
                    response += f"  [Watch on Crunchyroll]({anime['url']})\n\n"
                response += "💡 *Visit the AI Finder for more anime content!*"
                return response
            else:
                return f"🍥 No anime found for '{anime_query}'. Try popular anime like 'Demon Slayer', 'Jujutsu Kaisen', or 'Attack on Titan'."
        
        return "🍥 I can help you find anime! Try asking: 'Find anime Demon Slayer' or 'Search for Jujutsu Kaisen anime'"

    def _get_greeting_response(self, movie_data):
        """Personalized greeting based on user's collection"""
        total_movies = len(movie_data) if movie_data else 0
        watched_count = sum(1 for movie in movie_data if len(movie) > 4 and movie[4] == 1) if movie_data else 0
        
        greetings = [
            f"🎬 Welcome back, cinephile! I see you have {total_movies} movies in your collection ({watched_count} watched). I can help you search for movies, get details, and recommend films!",
            f"🌟 Hello there! With {total_movies} movies in your collection, we've got quite the film festival ahead! I can search for any movie you're curious about.",
            f"👋 Hey movie lover! Your collection of {total_movies} films is impressive! I can fetch detailed info or help you discover new movies.",
            f"🎭 Greetings, film enthusiast! {total_movies} movies and counting. I'm here with movie database integration to provide detailed information and recommendations!"
        ]
        
        return random.choice(greetings)

    def _get_recommendation_response(self, user_input, movie_data):
        """Intelligent movie recommendations"""
        if not movie_data:
            return "🎬 I'd love to recommend some movies! First, let's build your collection. You can also ask me to search for any movie, or try adding a few movies you enjoy!"
        
        # Get popular movies from enhanced database
        popular_movies = [m for m in MOVIE_DATABASE if m.get('streaming_service') or m.get('in_theaters')][:6]
        
        response = "🎯 **Popular Movies You Might Like**\n\n"
        
        for i, movie in enumerate(popular_movies, 1):
            streaming_info = ""
            if movie.get('streaming_service'):
                streaming_info = f" - 📺 {movie['streaming_service']}"
            elif movie.get('in_theaters'):
                streaming_info = " - 🎟️ In Theaters"
            
            response += f"{i}. **{movie['title']}** ({movie['year']}) - {movie['genre']}{streaming_info}\n"
        
        response += "\n🔍 *Use the AI Finder to search for these movies and add them to your collection!*"
        return response

    def _get_movie_details_response(self, user_input):
        """Get movie details from OMDB API based on user query"""
        words = user_input.split()
        movie_keywords = [word for word in words if word.lower() not in ['details', 'info', 'about', 'movie', 'get', 'tell', 'me']]
        
        if movie_keywords:
            movie_title = " ".join(movie_keywords[:4])
            
            with st.spinner(f"🔍 Searching for '{movie_title}'..."):
                movie_data = self.omdb.get_movie_details(movie_title)
            
            if movie_data:
                response = f"🎬 **{movie_data.get('Title', 'Movie')}** ({movie_data.get('Year', 'N/A')})\n\n"
                response += f"**Director:** {movie_data.get('Director', 'N/A')}\n"
                response += f"**Genre:** {movie_data.get('Genre', 'N/A')}\n"
                response += f"**Runtime:** {movie_data.get('Runtime', 'N/A')}\n"
                
                # Display ratings
                ratings = movie_data.get('Ratings', [])
                imdb_rating = movie_data.get('imdbRating', 'N/A')
                
                if imdb_rating != 'N/A':
                    response += f"**IMDB Rating:** {imdb_rating}/10\n"
                
                for rating in ratings:
                    source = rating.get('Source', '')
                    value = rating.get('Value', '')
                    if 'Rotten Tomatoes' in source:
                        response += f"**Rotten Tomatoes:** {value}\n"
                    elif 'Metacritic' in source:
                        response += f"**Metacritic:** {value}\n"
                
                response += f"**Plot:** {movie_data.get('Plot', 'N/A')}\n\n"
                
                response += f"💡 *Want to add this to your collection? Use the AI Finder feature!*"
                return response
            else:
                return f"❌ I couldn't find detailed information for '{movie_title}'. Try using the AI Finder feature for better results!"
        
        return "🎬 Please specify which movie you'd like details about! For example: 'Get details about Inception' or 'Tell me about The Dark Knight'"

    def _get_movie_search_response(self, user_input):
        """Handle movie search requests"""
        return "🔍 I'd be happy to help you search for movies! Use the AI Finder section to explore the database and find new movies to add to your collection."

    def _get_analysis_response(self, movie_data):
        """Analyze user's movie preferences and provide insights"""
        if not movie_data:
            return "📊 I'd love to analyze your movie taste! Start by adding some films to your collection, or use the AI Finder to discover and add new movies!"
        
        stats = get_stats()
        
        analysis = f"🎯 **Your Cinema Profile**\n\n"
        analysis += f"• **Collection Size**: {stats['total_movies']} films\n"
        analysis += f"• **Completion Rate**: {stats['watched_count']}/{stats['total_movies']} watched ({stats['completion_rate']:.1f}%)\n"
        analysis += f"• **In Theaters**: {stats['in_theaters_count']} movies\n"
        analysis += f"• **Average Rating**: {stats['average_rating']:.1f} ⭐\n"
        
        analysis += "\n🌟 **Recommendation**: Explore the AI Finder to discover more movies!"
        
        return analysis

    def _get_watch_status_response(self, movie_data):
        """Provide information about watched and unwatched movies"""
        if not movie_data:
            return "📝 Your collection is empty. Add some movies using the AI Finder feature to start tracking your watch progress!"
        
        watched_movies = [movie for movie in movie_data if len(movie) > 4 and movie[4] == 1]
        unwatched_movies = [movie for movie in movie_data if len(movie) > 4 and movie[4] == 0]
        
        response = f"📊 **Watch Status Overview**\n\n"
        response += f"• **Watched**: {len(watched_movies)} movies\n"
        response += f"• **Unwatched**: {len(unwatched_movies)} movies\n"
        response += f"• **Completion Rate**: {len(watched_movies)/len(movie_data)*100:.1f}%\n\n"
        
        if unwatched_movies:
            response += "🎬 **Top Unwatched Movies**:\n"
            for movie in unwatched_movies[:3]:
                response += f"• {movie[1]} ({movie[3]}) - {movie[2]}\n"
        
        response += f"\n🔍 *Find more movies to watch using AI Finder!*"
        
        return response

    def _get_help_response(self):
        """Comprehensive help guide"""
        return """
🤖 **Movie Assistant with Enhanced Features**

Here's what I can help you with:

🔍 **AI Finder Movie Search & Details**
• "Search for Inception on AI Finder"
• "Get details about The Dark Knight"
• "Find information about Parasite"

🎯 **Streaming & Watching**
• "Where can I watch Oppenheimer?"
• "Is Barbie on Netflix?"
• "Get tickets for Dune 2"

🍥 **Anime Content**
• "Find anime Demon Slayer"
• "Search for Jujutsu Kaisen"
• "Watch Attack on Titan on Crunchyroll"

📊 **Collection Management**
• "Analyze my movie taste"
• "What haven't I watched?"
• "My watchlist status"

🎬 **Recommendations**
• "Recommend action movies"
• "What should I watch tonight?"
• "Popular movies on Netflix"

💡 **Pro Tips**: 
• Use AI Finder to discover new movies
• Click "Watch Now" for streaming options
• Search for anime in AI Finder

What would you like to explore today?
        """

    def _get_best_movies_response(self, user_input):
        """Recommend best movies based on categories"""
        best_movies = {
            'all_time': [
                "The Godfather (1972) - Crime epic masterpiece",
                "The Shawshank Redemption (1994) - Ultimate story of hope",
                "The Dark Knight (2008) - Superhero cinema perfected",
                "Parasite (2019) - Brilliant social thriller",
                "Pulp Fiction (1994) - Revolutionary storytelling"
            ],
            'recent': [
                "Oppenheimer (2023) - Historical drama masterpiece",
                "Spider-Man: Across the Spider-Verse (2023) - Animation revolution",
                "Dune (2021) - Epic sci-fi spectacle",
                "Everything Everywhere All At Once (2022) - Multiverse madness",
                "The Batman (2022) - Dark detective thriller"
            ]
        }
        
        if 'recent' in user_input or 'new' in user_input:
            category = 'recent'
            title = "🎬 Best Recent Movies (2020s)"
        else:
            category = 'all_time'
            title = "🏆 All-Time Greatest Movies"
        
        response = f"{title}\n\n"
        for i, movie in enumerate(best_movies[category], 1):
            response += f"{i}. {movie}\n"
        
        response += f"\n🔍 *Search AI Finder for any of these movies to get detailed information!*"
        
        return response

    def _get_intelligent_fallback(self, user_input, movie_data):
        """Intelligent fallback for unexpected queries"""
        fallbacks = [
            f"🎬 That's an interesting question! I can help you search AI Finder for movies, get detailed information, or manage your collection. What would you like to know?",
            f"🤔 I'm not sure I understand completely. I'm here to help with movie searches, recommendations, and collection management using movie database data.",
            f"🔍 I specialize in movie information from databases and collection management. Try asking me to search for a movie, get details, or recommend something to watch!",
            f"🌟 Great question! I can fetch movie details from databases, help you discover new films, or analyze your collection. What movie-related topic can I assist with?"
        ]
        
        return random.choice(fallbacks)

    def _update_user_preferences(self, user_input, movie_data):
        """Learn from user interactions"""
        # Basic preference tracking
        genre_keywords = {
            'action': ['action', 'fight', 'adventure', 'thriller', 'exciting'],
            'comedy': ['comedy', 'funny', 'laugh', 'humor', 'hilarious'],
            'drama': ['drama', 'emotional', 'serious', 'story', 'deep'],
            'sci-fi': ['sci-fi', 'science fiction', 'future', 'space', 'alien'],
            'romance': ['romance', 'love', 'relationship', 'romantic', 'couple'],
            'anime': ['anime', 'manga', 'japanese animation']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                if genre not in self.user_preferences["favorite_genres"]:
                    self.user_preferences["favorite_genres"].append(genre)

# -----------------------------
# Enhanced Analytics Functions with Advanced Visualizations
# -----------------------------
def create_advanced_analytics_charts(movie_data):
    """Optimized chart creation with error handling"""
    if not movie_data:
        return tuple([None] * 6)
    
    # Convert to DataFrame once
    try:
        df = pd.DataFrame([
            {
                "title": m[1], "genre": m[2], "year": m[3],
                "watched": m[4] == 1, "rating": m[5] if len(m) > 5 else 0,
                "added_at": m[16] if len(m) > 16 and m[16] and not m[16].startswith('tt') 
                           else datetime.now().strftime('%Y-%m-%d')
            }
            for m in movie_data if len(m) >= 4
        ])
    except Exception as e:
        logging.error(f"DataFrame creation error: {e}")
        return tuple([None] * 6)
    
    if df.empty:
        return tuple([None] * 6)
    
    # Reusable chart config
    chart_config = {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font_color': 'white',
        'title_font_size': 20,
        'title_x': 0.5
    }
    
    charts = []
    
    # 1. Genre Distribution
    try:
        genre_counts = df['genre'].value_counts()
        fig = px.pie(values=genre_counts.values, names=genre_counts.index, 
                     title="🎭 Genre Distribution", hole=0.5)
        fig.update_layout(**chart_config)
        charts.append(fig)
    except Exception:
        charts.append(None)
    
    # 2. Watch Status
    try:
        watch_counts = df['watched'].value_counts()
        fig = px.bar(x=['Watched', 'Unwatched'], 
                     y=[watch_counts.get(True, 0), watch_counts.get(False, 0)], 
                     title="✅ Watch Status Overview",
                     color=['Watched', 'Unwatched'],
                     color_discrete_map={'Watched': '#6BCF7F', 'Unwatched': '#FF6B6B'})
        fig.update_layout(**chart_config)
        fig.update_traces(marker_line_color='white', marker_line_width=1.5)
        charts.append(fig)
    except Exception:
        charts.append(None)
    
    # 3. Year Distribution
    try:
        if 'year' in df.columns and df['year'].notna().any():
            year_counts = df['year'].value_counts().sort_index()
            # Filter out any non-numeric years
            year_counts = year_counts[year_counts.index.astype(str).str.isdigit()]
            if not year_counts.empty:
                fig = px.area(x=year_counts.index, y=year_counts.values, 
                              title="📅 Movie Collection Timeline",
                              color_discrete_sequence=['#FFD93D'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis_title="Release Year",
                    yaxis_title="Number of Movies",
                    title_font_size=20,
                    title_x=0.5
                )
                fig.update_traces(fillgradient=dict(
                    type="vertical",
                    colorscale=[[0, 'rgba(255,217,61,0.3)'], [1, 'rgba(255,217,61,0.8)']]
                ))
                charts.append(fig)
            else:
                charts.append(None)
        else:
            charts.append(None)
    except Exception:
        charts.append(None)
    
    # 4. Rating Distribution
    try:
        if 'rating' in df.columns and df['rating'].notna().any():
            rating_dist = df[df['rating'] > 0]['rating'].value_counts().sort_index()
            if not rating_dist.empty and len(rating_dist) > 1:
                fig = px.line_polar(r=rating_dist.values, theta=rating_dist.index, 
                                   title="⭐ User Ratings Distribution", line_close=True,
                                   color_discrete_sequence=['#FF6B6B'])
                fig.update_traces(fill='toself')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, color='white'),
                        angularaxis=dict(color='white')
                    ),
                    title_font_size=20,
                    title_x=0.5
                )
                charts.append(fig)
            else:
                charts.append(None)
        else:
            charts.append(None)
    except Exception:
        charts.append(None)
    
    # 5. Monthly Activity
    try:
        if 'added_at' in df.columns and df['added_at'].notna().any():
            # Safely convert to datetime
            valid_dates = []
            for date_str in df['added_at']:
                try:
                    if date_str and isinstance(date_str, str) and not date_str.startswith('tt'):
                        parsed_date = pd.to_datetime(date_str, errors='coerce')
                        if pd.notna(parsed_date):
                            valid_dates.append(parsed_date)
                        else:
                            valid_dates.append(pd.Timestamp.now())
                    else:
                        valid_dates.append(pd.Timestamp.now())
                except Exception:
                    valid_dates.append(pd.Timestamp.now())
            
            if valid_dates:
                temp_dates = pd.Series(valid_dates)
                monthly_activity = temp_dates.groupby(temp_dates.dt.to_period('M')).size()
                monthly_activity.index = monthly_activity.index.astype(str)
                
                if not monthly_activity.empty:
                    fig = px.bar(x=monthly_activity.index, y=monthly_activity.values,
                                 title="📈 Monthly Collection Growth",
                                 color_discrete_sequence=['#4D96FF'])
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        xaxis_title="Month",
                        yaxis_title="Movies Added",
                        title_font_size=20,
                        title_x=0.5
                    )
                    charts.append(fig)
                else:
                    charts.append(None)
            else:
                charts.append(None)
        else:
            charts.append(None)
    except Exception as e:
        print(f"Error creating monthly activity chart: {e}")
        charts.append(None)
    
    # 6. Genre vs Watch Status
    try:
        if not df.empty and 'genre' in df.columns and 'watched' in df.columns:
            genre_watch = pd.crosstab(df['genre'], df['watched'])
            if 'Unwatched' not in genre_watch.columns:
                genre_watch['Unwatched'] = 0
            if 'Watched' not in genre_watch.columns:
                genre_watch['Watched'] = 0
                
            genre_watch.columns = ['Unwatched', 'Watched']
            
            if not genre_watch.empty:
                fig = px.bar(genre_watch, x=genre_watch.index, y=['Watched', 'Unwatched'],
                             title="🎯 Genre Completion Analysis",
                             color_discrete_map={'Watched': '#6BCF7F', 'Unwatched': '#FF6B6B'})
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis_title="Genre",
                    yaxis_title="Number of Movies",
                    title_font_size=20,
                    title_x=0.5,
                    barmode='stack'
                )
                charts.append(fig)
            else:
                charts.append(None)
        else:
            charts.append(None)
    except Exception:
        charts.append(None)
    
    return tuple(charts)

# -----------------------------
# Enhanced Analytics Page
# -----------------------------
def show_analytics_page(movies_data):
    """Enhanced Analytics page with cinematic styling and error handling"""
    st.markdown("""
    <div class="analytics-hero">
        <h1 style="text-align: center; font-size: 3.5rem; margin-bottom: 1rem;">📊 CINEMA ANALYTICS</h1>
        <p style="text-align: center; font-size: 1.5rem; color: #FFD93D; margin: 0;">
            Advanced Insights & Visual Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not movies_data:
        st.info("🎬 Add some movies to your collection to unlock advanced analytics!")
        return
    
    stats = get_stats()
    
    # Enhanced Metrics Grid
    st.markdown("### 🎯 REAL-TIME CINEMA METRICS")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="analytics-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">🎬</div>
                <div class="stat-number">{stats['total_movies']}</div>
                <div class="stat-label">Total Movies</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="analytics-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">✅</div>
                <div class="stat-number">{stats['watched_count']}</div>
                <div class="stat-label">Movies Watched</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="analytics-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">📈</div>
                <div class="stat-number">{stats['completion_rate']:.1f}%</div>
                <div class="stat-label">Completion Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="analytics-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">⭐</div>
                <div class="stat-number">{stats['average_rating']:.1f}</div>
                <div class="stat-label">Avg Rating</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Charts Section
    st.markdown("### 📈 ADVANCED VISUAL ANALYTICS")
    
    # Generate all charts with error handling using cached version
    try:
        # Create hash for caching
        import hashlib
        movies_json = json.dumps([m[0] for m in movies_data])  # Just IDs
        data_hash = hashlib.md5(movies_json.encode()).hexdigest()
        charts = create_analytics_charts_cached(data_hash)
    except Exception as e:
        st.error(f"Error generating analytics charts: {e}")
        st.info("Some charts may not be available due to data issues.")
        charts = tuple([None] * 6)
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        if charts[0]:
            st.plotly_chart(charts[0], use_container_width=True, key="genre_distribution_chart")
        else:
            st.info("Genre distribution chart not available")
    
    with col2:
        if charts[1]:
            st.plotly_chart(charts[1], use_container_width=True, key="watch_status_chart")
        else:
            st.info("Watch status chart not available")
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        if charts[2]:
            st.plotly_chart(charts[2], use_container_width=True, key="timeline_chart")
        else:
            st.info("Timeline chart not available")
    
    with col4:
        if charts[3]:
            st.plotly_chart(charts[3], use_container_width=True, key="ratings_radar_chart")
        else:
            st.info("Ratings distribution chart not available")
    
    # Third row of charts
    if charts[4] or charts[5]:
        col5, col6 = st.columns(2)
        
        with col5:
            if charts[4]:
                st.plotly_chart(charts[4], use_container_width=True, key="monthly_activity_chart")
            else:
                st.info("Monthly activity chart not available")
        
        with col6:
            if charts[5]:
                st.plotly_chart(charts[5], use_container_width=True, key="genre_completion_chart")
            else:
                st.info("Genre completion chart not available")
    
    # Detailed Statistics Section
    st.markdown("### 🔍 DETAILED STATISTICS")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown("""
        <div class="insight-card">
            <h4>🎭 Genre Diversity</h4>
            <p>Your collection spans <strong>{}</strong> unique genres, showing {} taste in cinema.</p>
        </div>
        """.format(
            stats['unique_genres'],
            "diverse" if stats['unique_genres'] > 5 else "focused"
        ), unsafe_allow_html=True)
    
    with col8:
        watch_ratio = stats['watched_count'] / stats['total_movies'] if stats['total_movies'] > 0 else 0
        status = "Highly Active" if watch_ratio > 0.7 else "Moderate" if watch_ratio > 0.4 else "Casual"
        st.markdown("""
        <div class="insight-card">
            <h4>⏱️ Watch Behavior</h4>
            <p>You've watched <strong>{:.1f}%</strong> of your collection - {} viewer profile.</p>
        </div>
        """.format(stats['completion_rate'], status), unsafe_allow_html=True)
    
    with col9:
        theater_ratio = stats['in_theaters_count'] / stats['total_movies'] if stats['total_movies'] > 0 else 0
        trend = "Trending" if theater_ratio > 0.3 else "Classic" if theater_ratio > 0.1 else "Vintage"
        st.markdown("""
        <div class="insight-card">
            <h4>🎟️ Content Freshness</h4>
            <p><strong>{}</strong> recent releases - {} collection focus.</p>
        </div>
        """.format(stats['in_theaters_count'], trend), unsafe_allow_html=True)
    
    # Recommendations based on analytics
    st.markdown("### 💡 INTELLIGENT RECOMMENDATIONS")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        if stats['completion_rate'] < 50:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #FF6B6B;">
                <h4>🚀 Boost Your Progress</h4>
                <p>Try watching 2-3 movies from your unwatched list this week!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #6BCF7F;">
                <h4>🎯 Great Progress!</h4>
                <p>You're maintaining excellent watch habits. Keep it up!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with rec_col2:
        if stats['unique_genres'] < 4:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #FFD93D;">
                <h4>🎭 Expand Horizons</h4>
                <p>Explore new genres to diversify your movie experience!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #6BCF7F;">
                <h4>🌈 Diverse Tastes</h4>
                <p>Your genre variety shows great cinematic appreciation!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with rec_col3:
        if stats['average_rating'] < 3:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #FF6B6B;">
                <h4>⭐ Rating Insight</h4>
                <p>Consider exploring highly-rated movies to enhance enjoyment!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #6BCF7F;">
                <h4>👍 Quality Focus</h4>
                <p>You're great at picking movies you enjoy. Excellent taste!</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# New Streaming & Ticketing Page
# -----------------------------
def show_streaming_page(movies_data):
    """Dedicated page for streaming and ticketing options"""
    st.markdown("""
    <div class="hero-container">
        <h2 style="text-align: center; font-size: 2.5rem; margin-bottom: 1rem;">🎯 WATCH NOW</h2>
        <p style="text-align: center; font-size: 1.3rem; color: #FFD93D; margin: 0;">
            Streaming & Ticketing Across All Platforms
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick access section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎟️ IN THEATERS NOW", use_container_width=True, type="primary"):
            st.session_state.streaming_filter = "theaters"
            st.rerun()
    
    with col2:
        if st.button("📺 POPULAR ON STREAMING", use_container_width=True, type="primary"):
            st.session_state.streaming_filter = "streaming"
            st.rerun()
    
    with col3:
        if st.button("🔍 FIND ANY MOVIE", use_container_width=True, type="primary"):
            st.session_state.streaming_filter = "search"
            st.rerun()
    
    # Filter section
    st.markdown("### 🎬 YOUR WATCHABLE MOVIES")
    
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        search_query = st.text_input("🔍 Search movies to watch...", placeholder="Enter movie title...")
    with filter_col2:
        availability_filter = st.selectbox("Availability", ["All", "In Theaters", "On Streaming"])
    
    # Filter movies
    filtered_movies = movies_data
    if search_query:
        filtered_movies = [m for m in filtered_movies if search_query.lower() in m[1].lower()]
    
    if availability_filter == "In Theaters":
        filtered_movies = [m for m in filtered_movies if len(m) > 15 and m[15] == 1]
    elif availability_filter == "On Streaming":
        # For demo, consider movies from last 5 years as likely on streaming
        current_year = datetime.now().year
        filtered_movies = [m for m in filtered_movies if len(m) > 3 and m[3] <= current_year - 1]
    
    if filtered_movies:
        st.markdown(f"### FOUND {len(filtered_movies)} MOVIES")
        
        for i, movie in enumerate(filtered_movies):
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    title = movie[1]
                    year = movie[3]
                    genre = movie[2]
                    
                    st.markdown(f"#### {title} ({year})")
                    st.markdown(f"**Genre:** {genre}")
                    
                    # Get OMDB details for ratings
                    omdb = RateLimitedOMDbAPI()
                    movie_details = omdb.get_movie_details(title, year)
                    
                    if movie_details:
                        imdb_rating = movie_details.get('imdbRating', 'N/A')
                        if imdb_rating != 'N/A':
                            st.markdown(f'<span class="rating-badge">⭐ IMDB: {imdb_rating}/10</span>', unsafe_allow_html=True)
                        
                        # Check for Rotten Tomatoes
                        ratings = movie_details.get('Ratings', [])
                        for rating in ratings:
                            if 'Rotten Tomatoes' in rating.get('Source', ''):
                                st.markdown(f'<span style="color: #FF6B6B">🍅 {rating.get("Value", "")}</span>', unsafe_allow_html=True)
                    
                    # Quick streaming check
                    streaming_finder = EnhancedStreamingServiceFinder()
                    watch_options = streaming_finder.get_watch_options(title, year, genre)
                    
                    available_streaming = [s for s in watch_options['streaming'].values() if s['available']]
                    in_theaters = watch_options['in_theaters']
                    
                    if available_streaming:
                        st.markdown("**Streaming:** " + ", ".join([s['name'] for s in available_streaming[:2]]))
                    if in_theaters:
                        st.markdown("🎟️ **In Theaters**")
                
                with col2:
                    if movie[8] and movie[8] != "N/A":
                        st.image(movie[8], width=80)
                
                with col3:
                    if st.button("🎯 WATCH OPTIONS", key=f"stream_{movie[0]}_{i}", use_container_width=True, type="primary"):
                        st.session_state.app_state.show_watch_options.add(movie[0])
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show watch options if triggered
                if movie[0] in st.session_state.app_state.show_watch_options:
                    display_watch_options_section(movie[1], movie[3], movie[2], movie[0], f"stream_{i}")
    else:
        st.info("🎬 No movies found matching your criteria. Try adding more movies to your collection!")

# -----------------------------
# Helper Functions
# -----------------------------
def add_sample_movies(count=5):
    """Add sample movies to the collection"""
    added_count = 0
    existing_movies = get_movies_safe()
    existing_titles = [movie[1].lower() for movie in existing_movies] if existing_movies else []
    
    available_movies = [movie for movie in MOVIE_DATABASE 
                       if movie["title"].lower() not in existing_titles]
    
    if not available_movies:
        return 0
    
    movies_to_add = random.sample(available_movies, min(count, len(available_movies)))
    
    for movie in movies_to_add:
        try:
            add_movie(title=movie["title"], genre=movie["genre"], year=movie["year"], watched=random.choice([True, False]))
            added_count += 1
        except Exception as e:
            logging.error(f"Error adding sample movie {movie['title']}: {e}")
            continue
    
    return added_count

# -----------------------------
# Page Functions
# -----------------------------
def show_dashboard(movies_data, stats):
    """Ultra Enhanced Dashboard with Cinematic Animations"""
    
    # Cinematic Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="cinematic-title">CODEFLIX</h1>
        <p class="cinematic-subtitle">Your Personal Cinematic Universe</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎬 CINEMA INTELLIGENCE DASHBOARD")
    
    # Enhanced Metrics with Holographic Effect
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <div style='font-size: 2.5rem; animation: cinematicPulse 2s infinite;'>🎬</div>
            <h3 style='margin: 10px 0;'>Total Movies</h3>
            <p style='font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #FF6B6B, #FFD93D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>{stats['total_movies']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <div style='font-size: 2.5rem; animation: cinematicPulse 2s infinite;'>✅</div>
            <h3 style='margin: 10px 0;'>Movies Watched</h3>
            <p style='font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6BCF7F, #4CAF50); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>{stats['watched_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <div style='font-size: 2.5rem; animation: cinematicPulse 2s infinite;'>🎟️</div>
            <h3 style='margin: 10px 0;'>In Theaters</h3>
            <p style='font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #FF6B00, #FF8C00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>{stats['in_theaters_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col4:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <div style='font-size: 2.5rem; animation: cinematicPulse 2s infinite;'>🎭</div>
            <h3 style='margin: 10px 0;'>Unique Genres</h3>
            <p style='font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #9D4BFF, #6B5AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>{stats['unique_genres']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Watch Section with Enhanced Animation
    st.markdown("### ⚡ QUICK WATCH OPTIONS")
    
    if movies_data:
        # Show recent movies with enhanced watch buttons
        recent_movies = movies_data[:3]
        cols = st.columns(3)
        
        for idx, movie in enumerate(recent_movies):
            with cols[idx]:
                title = movie[1]
                year = movie[3]
                
                st.markdown(f"**{title}** ({year})")
                if movie[8] and movie[8] != "N/A":
                    st.image(movie[8], width="stretch")
                else:
                    st.markdown("🎭 *No poster available*")
                
                if st.button("🎯 WATCH NOW", key=f"quick_{movie[0]}", use_container_width=True, type="primary"):
                    st.session_state.app_state.show_watch_options.add(movie[0])
                    st.rerun()
        
        # Show watch options if any were triggered
        for movie in recent_movies:
            if movie[0] in st.session_state.app_state.show_watch_options:
                display_watch_options_section(movie[1], movie[3], movie[2], movie[0], "dashboard")
    else:
        st.info("🎬 Your collection is empty! Add some movies to get started.")
        
        # Show sample movies to add
        st.markdown("### 🚀 GET STARTED WITH THESE POPULAR MOVIES")
        sample_movies = [m for m in MOVIE_DATABASE if m.get('streaming_service')][:6]
        sample_cols = st.columns(3)
        
        for idx, movie in enumerate(sample_movies):
            with sample_cols[idx % 3]:
                st.markdown(f"**{movie['title']}** ({movie['year']})")
                st.markdown(f"*{movie['genre']}*")
                if st.button("Add", key=f"sample_{idx}", use_container_width=True, type="primary"):
                    add_movie(movie['title'], movie['genre'], movie['year'], False)
                    st.success(f"Added {movie['title']}!")
                    st.rerun()

def show_add_movies_page():
    """Add movies page with enhanced design"""
    st.markdown("""
    <div class="hero-container">
        <h2 style="text-align: center; font-size: 2.5rem; margin-bottom: 1rem;">🎬 ADD NEW MOVIES</h2>
        <p style="text-align: center; font-size: 1.3rem; color: #FFD93D; margin: 0;">
            Expand Your Cinematic Collection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("add_movie_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("🎬 Movie Title", placeholder="Enter movie title...")
            genre = st.selectbox("🎭 Genre", ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Horror", "Adventure", "Fantasy", "Animation", "Documentary", "Mystery", "Anime"])
        with col2:
            year = st.number_input("📅 Release Year", min_value=1900, max_value=2030, value=2023)
            watched = st.checkbox("✅ Watched")
            rating = st.slider("⭐ Your Rating", 0, 5, 0, help="0 means not rated")
        
        review = st.text_area("📝 Your Review (optional)", placeholder="Share your thoughts about this movie...")
        
        submitted = st.form_submit_button("🎬 ADD MOVIE", type="primary")
        
        if submitted and title.strip():
            errors = validate_movie_data(title, genre, year)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    add_movie(
                        title=sanitize_input(title), 
                        genre=genre, 
                        year=year, 
                        watched=watched,
                        rating=rating,
                        review=sanitize_input(review)
                    )
                    st.success(f"✅ '{title}' added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding movie: {e}")
    
    # Quick add from popular movies with enhanced design
    st.markdown("### 🚀 QUICK ADD POPULAR MOVIES")
    popular_movies = [m for m in MOVIE_DATABASE if m.get('streaming_service') or m.get('in_theaters')][:9]
    cols = st.columns(3)
    for idx, movie in enumerate(popular_movies):
        with cols[idx % 3]:
            streaming_info = ""
            if movie.get('streaming_service'):
                streaming_info = f"📺 {movie['streaming_service']}"
            elif movie.get('in_theaters'):
                streaming_info = "🎟️ In Theaters"
            
            st.markdown(f"**{movie['title']}** ({movie['year']})")
            st.markdown(f"*{movie['genre']}*")
            st.markdown(f"*{streaming_info}*")
            if st.button("Add", key=f"quick_add_{idx}", use_container_width=True, type="primary"):
                add_movie(movie['title'], movie['genre'], movie['year'], False)
                st.success(f"Added {movie['title']}!")
                st.rerun()

def show_collection_page(movies_data):
    """My Collection page with enhanced filtering"""
    st.markdown("""
    <div class="hero-container">
        <h2 style="text-align: center; font-size: 2.5rem; margin-bottom: 1rem;">🎬 MY MOVIE COLLECTION</h2>
        <p style="text-align: center; font-size: 1.3rem; color: #FFD93D; margin: 0;">
            Your Personal Cinema Library
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not movies_data:
        st.info("🎬 Your collection is empty! Add some movies to get started.")
        return
    
    # Enhanced filtering with modern design
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search_term = st.text_input("🔍 Search movies...", placeholder="Search by title, genre, director...")
    with col2:
        all_genres = list(set(movie[2] for movie in movies_data if len(movie) > 2))
        genre_filter = st.selectbox("🎭 Genre", ["All"] + all_genres)
    with col3:
        watched_filter = st.selectbox("✅ Status", ["All", "Watched", "Unwatched"])
    with col4:
        rating_filter = st.selectbox("⭐ Rating", ["All", "1+ Stars", "2+ Stars", "3+ Stars", "4+ Stars", "5 Stars"])
    
    # Apply filters
    filters = {}
    if search_term:
        filters['search'] = search_term
    if genre_filter != "All":
        filters['genre'] = genre_filter
    if watched_filter != "All":
        filters['watched'] = watched_filter
    
    # Get paginated data
    app_state = st.session_state.app_state
    page_size = 10
    paginated_movies, total_count = get_movies_paginated(app_state.current_page, page_size, filters)
    
    # Pagination
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ PREVIOUS", disabled=app_state.current_page == 0, use_container_width=True):
            app_state.current_page -= 1
            st.rerun()
    with col_page:
        st.markdown(f"**Page {app_state.current_page + 1} of {total_pages}**")
    with col_next:
        if st.button("NEXT ➡️", disabled=app_state.current_page >= total_pages - 1, use_container_width=True):
            app_state.current_page += 1
            st.rerun()
    
    # Apply rating filter
    if rating_filter != "All":
        min_rating = int(rating_filter[0])
        paginated_movies = [m for m in paginated_movies if len(m) > 5 and m[5] >= min_rating]
    
    if paginated_movies:
        st.markdown(f"### 📊 FOUND {len(paginated_movies)} MOVIES")
        for i, movie in enumerate(paginated_movies):
            create_movie_card(movie, key_suffix=f"collection_{i}")
    else:
        st.info("🎬 No movies found matching your filters.")

# -----------------------------
# Input Validation and Error Handling
# -----------------------------
def sanitize_input(text):
    """Basic input sanitization"""
    if not text:
        return ""
    return html.escape(text.strip())

def validate_movie_data(title, genre, year):
    """Validate movie data before adding to database"""
    errors = []
    
    if not title or len(title.strip()) == 0:
        errors.append("Movie title is required")
    
    if not genre or len(genre.strip()) == 0:
        errors.append("Genre is required")
    
    current_year = datetime.now().year
    if year < 1900 or year > current_year + 5:
        errors.append(f"Year must be between 1900 and {current_year + 5}")
    
    return errors

# -----------------------------
# ColorBends Background Animation Component
# -----------------------------
def inject_color_bends_background():
    """Inject the ColorBends animated background"""
    st.markdown("""
    <style>
    /* ColorBends Background Container */
    .color-bends-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -9999;
        pointer-events: none;
        opacity: 0.7;
    }
    
    /* Ensure main content is above background */
    .main .block-container {
        position: relative;
        z-index: 1;
        background: rgba(12, 12, 12, 0.85);
        border-radius: 15px;
        margin: 1rem;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background: rgba(12, 12, 12, 0.9) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    </style>
    
    <div class="color-bends-background" id="colorBendsContainer"></div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    class ColorBendsBackground {
        constructor() {
            this.container = document.getElementById('colorBendsContainer');
            this.scene = new THREE.Scene();
            this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
            this.renderer = new THREE.WebGLRenderer({ 
                antialias: false, 
                alpha: true,
                powerPreference: 'high-performance'
            });
            
            this.setupRenderer();
            this.createGeometry();
            this.setupShader();
            this.animate();
            this.handleResize();
            
            window.addEventListener('resize', () => this.handleResize());
        }
        
        setupRenderer() {
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.renderer.setClearColor(0x000000, 0);
            this.renderer.domElement.style.width = '100%';
            this.renderer.domElement.style.height = '100%';
            this.renderer.domElement.style.display = 'block';
            this.container.appendChild(this.renderer.domElement);
        }
        
        createGeometry() {
            this.geometry = new THREE.PlaneGeometry(2, 2);
        }
        
        setupShader() {
            const MAX_COLORS = 8;
            
            const vertexShader = `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = vec4(position, 1.0);
                }
            `;
            
            const fragmentShader = `
                #define MAX_COLORS ${MAX_COLORS}
                uniform vec2 uCanvas;
                uniform float uTime;
                uniform float uSpeed;
                uniform vec2 uRot;
                uniform int uColorCount;
                uniform vec3 uColors[MAX_COLORS];
                uniform int uTransparent;
                uniform float uScale;
                uniform float uFrequency;
                uniform float uWarpStrength;
                uniform vec2 uPointer;
                uniform float uMouseInfluence;
                uniform float uParallax;
                uniform float uNoise;
                varying vec2 vUv;

                void main() {
                    float t = uTime * uSpeed;
                    vec2 p = vUv * 2.0 - 1.0;
                    p += uPointer * uParallax * 0.1;
                    vec2 rp = vec2(p.x * uRot.x - p.y * uRot.y, p.x * uRot.y + p.y * uRot.x);
                    vec2 q = vec2(rp.x * (uCanvas.x / uCanvas.y), rp.y);
                    q /= max(uScale, 0.0001);
                    q /= 0.5 + 0.2 * dot(q, q);
                    q += 0.2 * cos(t) - 7.56;
                    vec2 toward = (uPointer - rp);
                    q += toward * uMouseInfluence * 0.2;

                    vec3 col = vec3(0.0);
                    float a = 1.0;

                    if (uColorCount > 0) {
                        vec2 s = q;
                        vec3 sumCol = vec3(0.0);
                        float cover = 0.0;
                        for (int i = 0; i < MAX_COLORS; ++i) {
                            if (i >= uColorCount) break;
                            s -= 0.01;
                            vec2 r = sin(1.5 * (s.yx * uFrequency) + 2.0 * cos(s * uFrequency));
                            float m0 = length(r + sin(5.0 * r.y * uFrequency - 3.0 * t + float(i)) / 4.0);
                            float kBelow = clamp(uWarpStrength, 0.0, 1.0);
                            float kMix = pow(kBelow, 0.3);
                            float gain = 1.0 + max(uWarpStrength - 1.0, 0.0);
                            vec2 disp = (r - s) * kBelow;
                            vec2 warped = s + disp * gain;
                            float m1 = length(warped + sin(5.0 * warped.y * uFrequency - 3.0 * t + float(i)) / 4.0);
                            float m = mix(m0, m1, kMix);
                            float w = 1.0 - exp(-6.0 / exp(6.0 * m));
                            sumCol += uColors[i] * w;
                            cover = max(cover, w);
                        }
                        col = clamp(sumCol, 0.0, 1.0);
                        a = uTransparent > 0 ? cover : 1.0;
                    } else {
                        vec2 s = q;
                        for (int k = 0; k < 3; ++k) {
                            s -= 0.01;
                            vec2 r = sin(1.5 * (s.yx * uFrequency) + 2.0 * cos(s * uFrequency));
                            float m0 = length(r + sin(5.0 * r.y * uFrequency - 3.0 * t + float(k)) / 4.0);
                            float kBelow = clamp(uWarpStrength, 0.0, 1.0);
                            float kMix = pow(kBelow, 0.3);
                            float gain = 1.0 + max(uWarpStrength - 1.0, 0.0);
                            vec2 disp = (r - s) * kBelow;
                            vec2 warped = s + disp * gain;
                            float m1 = length(warped + sin(5.0 * warped.y * uFrequency - 3.0 * t + float(k)) / 4.0);
                            float m = mix(m0, m1, kMix);
                            col[k] = 1.0 - exp(-6.0 / exp(6.0 * m));
                        }
                        a = uTransparent > 0 ? max(max(col.r, col.g), col.b) : 1.0;
                    }

                    if (uNoise > 0.0001) {
                        float n = fract(sin(dot(gl_FragCoord.xy + vec2(uTime), vec2(12.9898, 78.233))) * 43758.5453123);
                        col += (n - 0.5) * uNoise;
                        col = clamp(col, 0.0, 1.0);
                    }

                    vec3 rgb = (uTransparent > 0) ? col * a : col;
                    gl_FragColor = vec4(rgb, a * 0.3);
                }
            `;
            
            // Enhanced color palette for movie theme
            const colors = [
                new THREE.Vector3(1.0, 0.36, 0.48),  // #ff5c7a - Pink
                new THREE.Vector3(0.54, 0.36, 1.0),  // #8a5cff - Purple  
                new THREE.Vector3(0.0, 1.0, 0.82),   // #00ffd1 - Cyan
                new THREE.Vector3(1.0, 0.85, 0.24),  // #ffd93d - Yellow
                new THREE.Vector3(0.42, 0.81, 0.5),  // #6bcf7f - Green
                new THREE.Vector3(0.3, 0.59, 1.0)    // #4d96ff - Blue
            ];
            
            this.uColorsArray = Array.from({ length: MAX_COLORS }, (_, i) => 
                colors[i] || new THREE.Vector3(0, 0, 0)
            );
            
            this.uniforms = {
                uCanvas: { value: new THREE.Vector2(this.container.clientWidth, this.container.clientHeight) },
                uTime: { value: 0 },
                uSpeed: { value: 0.3 },
                uRot: { value: new THREE.Vector2(Math.cos(0.5), Math.sin(0.5)) },
                uColorCount: { value: colors.length },
                uColors: { value: this.uColorsArray },
                uTransparent: { value: 1 },
                uScale: { value: 1.2 },
                uFrequency: { value: 1.4 },
                uWarpStrength: { value: 1.2 },
                uPointer: { value: new THREE.Vector2(0, 0) },
                uMouseInfluence: { value: 0.8 },
                uParallax: { value: 0.6 },
                uNoise: { value: 0.08 }
            };
            
            this.material = new THREE.ShaderMaterial({
                vertexShader: vertexShader,
                fragmentShader: fragmentShader,
                uniforms: this.uniforms,
                transparent: true
            });
            
            this.mesh = new THREE.Mesh(this.geometry, this.material);
            this.scene.add(this.mesh);
            
            // Mouse interaction
            this.mouse = new THREE.Vector2(0, 0);
            this.targetMouse = new THREE.Vector2(0, 0);
            document.addEventListener('mousemove', (e) => {
                this.targetMouse.x = (e.clientX / window.innerWidth) * 2 - 1;
                this.targetMouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
            });
        }
        
        animate() {
            requestAnimationFrame(() => this.animate());
            
            this.uniforms.uTime.value += 0.016;
            
            // Smooth mouse follow
            this.mouse.lerp(this.targetMouse, 0.05);
            this.uniforms.uPointer.value.copy(this.mouse);
            
            // Auto rotation
            const time = this.uniforms.uTime.value * 0.1;
            const rotation = time * 0.3;
            this.uniforms.uRot.value.set(Math.cos(rotation), Math.sin(rotation));
            
            this.renderer.render(this.scene, this.camera);
        }
        
        handleResize() {
            const width = this.container.clientWidth;
            const height = this.container.clientHeight;
            
            this.renderer.setSize(width, height);
            this.uniforms.uCanvas.value.set(width, height);
        }
    }
    
    // Initialize when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => new ColorBendsBackground());
    } else {
        new ColorBendsBackground();
    }
    </script>
    """, unsafe_allow_html=True)

# -----------------------------
# Ultra Advanced Custom CSS with Cinematic Animations
# -----------------------------
def inject_advanced_style():
    """Inject enhanced CSS with ColorBends integration"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: transparent !important;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        min-height: 100vh;
    }
    
    /* Main content area with glass effect */
    .main .block-container {
        background: rgba(12, 12, 12, 0.85) !important;
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    
    /* Cinematic Floating Particles */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -9998;
    }
    
    .particle {
        position: absolute;
        background: radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        animation: float 6s infinite linear;
    }
    
    @keyframes float {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(-100px) rotate(360deg); opacity: 0; }
    }
    
    /* Ultra Animated CodeFlix Title */
    .cinematic-title {
        font-family: 'Orbitron', monospace;
        font-size: 5.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, 
            #FF6B6B, #FFD93D, #6BCF7F, #4D96FF, 
            #9D4BFF, #FF6B6B, #FFD93D, #6BCF7F);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: rainbowGlow 8s ease infinite, cinematicPulse 3s ease-in-out infinite;
        text-shadow: 
            0 0 20px rgba(255, 107, 107, 0.3),
            0 0 40px rgba(255, 217, 61, 0.2),
            0 0 60px rgba(109, 207, 127, 0.1);
        margin: 0;
        padding: 1rem 0;
        position: relative;
        letter-spacing: 2px;
    }
    
    .cinematic-title::before {
        content: 'CODEFLIX';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, 
            transparent 45%, 
            rgba(255, 255, 255, 0.1) 50%, 
            transparent 55%);
        background-size: 200% 100%;
        animation: shimmer 4s infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    @keyframes rainbowGlow {
        0%, 100% { background-position: 0% 50%; filter: hue-rotate(0deg); }
        50% { background-position: 100% 50%; filter: hue-rotate(180deg); }
    }
    
    @keyframes cinematicPulse {
        0%, 100% { transform: scale(1); text-shadow: 0 0 20px rgba(255, 107, 107, 0.3); }
        50% { transform: scale(1.02); text-shadow: 0 0 40px rgba(255, 217, 61, 0.4); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .cinematic-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        text-align: center;
        background: linear-gradient(135deg, #FFD93D, #6BCF7F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: subtitleGlow 3s ease-in-out infinite alternate;
        margin: 0;
        padding-bottom: 3rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    @keyframes subtitleGlow {
        0% { text-shadow: 0 0 10px rgba(255, 217, 61, 0.3); }
        100% { text-shadow: 0 0 20px rgba(107, 207, 127, 0.5); }
    }
    
    .hero-container {
        background: linear-gradient(135deg, 
            rgba(12, 12, 12, 0.95) 0%, 
            rgba(26, 26, 46, 0.9) 50%, 
            rgba(15, 52, 96, 0.85) 100%);
        padding: 4rem 3rem;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 3rem;
        backdrop-filter: blur(20px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.5),
            0 0 100px rgba(77, 150, 255, 0.1);
        position: relative;
        overflow: hidden;
        animation: containerFloat 6s ease-in-out infinite;
    }
    
    @keyframes containerFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            from 0deg,
            transparent,
            rgba(255, 107, 107, 0.1),
            rgba(255, 217, 61, 0.1),
            rgba(107, 207, 127, 0.1),
            rgba(77, 150, 255, 0.1),
            transparent
        );
        animation: rotate 10s linear infinite;
        z-index: -1;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Glass Cards */
    .glass-card {
        background: rgba(20, 20, 35, 0.7);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.1),
            transparent
        );
        transition: left 0.6s;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(255, 107, 107, 0.5);
        box-shadow: 
            0 25px 50px rgba(255, 107, 107, 0.2),
            0 0 80px rgba(255, 107, 107, 0.1);
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    /* Advanced Buttons */
    .advanced-btn {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 50%, #6BCF7F 100%) !important;
        background-size: 200% 200% !important;
        color: #0c0c0c !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px 30px !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        font-family: 'Rajdhani', sans-serif !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 8px 25px rgba(255, 107, 107, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        position: relative;
        overflow: hidden;
        letter-spacing: 1px;
    }
    
    .advanced-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.6s;
    }
    
    .advanced-btn:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 
            0 15px 35px rgba(255, 107, 107, 0.6),
            0 0 40px rgba(255, 217, 61, 0.4) !important;
        background-position: 100% 100% !important;
    }
    
    .advanced-btn:hover::before {
        left: 100%;
    }
    
    /* Movie Cards with Holographic Effect */
    .movie-card {
        background: linear-gradient(135deg, 
            rgba(30, 30, 45, 0.9), 
            rgba(20, 20, 35, 0.9));
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, 
            #FF6B6B, #FFD93D, #6BCF7F, #4D96FF, #9D4BFF);
        background-size: 400% 400%;
        z-index: -1;
        border-radius: 22px;
        animation: borderGlow 6s linear infinite;
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .movie-card:hover::before {
        opacity: 1;
    }
    
    .movie-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: transparent;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(255, 107, 107, 0.2);
    }
    
    @keyframes borderGlow {
        0%, 100% { background-position: 0% 50%; filter: hue-rotate(0deg); }
        50% { background-position: 100% 50%; filter: hue-rotate(180deg); }
    }
    
    /* Enhanced Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #FFFFFF, #FFD93D, #6BCF7F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Movie Detail Cards */
    .movie-detail-card {
        background: linear-gradient(135deg, 
            rgba(25, 25, 40, 0.95), 
            rgba(15, 15, 30, 0.95));
        border-radius: 20px;
        padding: 30px;
        margin: 25px 0;
        border: 1px solid rgba(255, 107, 107, 0.3);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .movie-detail-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #FF6B6B, #FFD93D, #6BCF7F);
        background-size: 200% 100%;
        animation: loadingBar 3s ease-in-out infinite;
    }
    
    @keyframes loadingBar {
        0%, 100% { background-position: -200% 0; }
        50% { background-position: 200% 0; }
    }
    
    /* Rating Badges */
    .rating-badge {
        background: linear-gradient(135deg, #FFD93D, #FF6B6B);
        color: #0c0c0c !important;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 800;
        font-size: 0.9em;
        font-family: 'Rajdhani', sans-serif;
        box-shadow: 0 4px 15px rgba(255, 217, 61, 0.3);
        animation: badgePulse 2s ease-in-out infinite;
    }
    
    @keyframes badgePulse {
        0%, 100% { transform: scale(1); box-shadow: 0 4px 15px rgba(255, 217, 61, 0.3); }
        50% { transform: scale(1.05); box-shadow: 0 6px 20px rgba(255, 217, 61, 0.5); }
    }
    
    /* Star Rating */
    .star-rating {
        color: #FFD93D;
        font-size: 1.3em;
        text-shadow: 0 0 10px rgba(255, 217, 61, 0.5);
        animation: starTwinkle 3s ease-in-out infinite;
    }
    
    @keyframes starTwinkle {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Streaming Badges */
    .streaming-badge {
        display: inline-block;
        padding: 10px 20px;
        margin: 6px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9em;
        font-family: 'Rajdhani', sans-serif;
        transition: all 0.3s ease;
        text-decoration: none;
        border: 2px solid transparent;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .streaming-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .streaming-badge:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .streaming-badge:hover::before {
        left: 100%;
    }
    
    .available-service {
        background: linear-gradient(135deg, #6BCF7F, #4CAF50);
        color: white;
    }
    
    .unavailable-service {
        background: rgba(255, 255, 255, 0.1);
        color: #888;
    }
    
    .ticket-service {
        background: linear-gradient(135deg, #FF6B00, #FF8C00);
        color: white;
    }
    
    .anime-service {
        background: linear-gradient(135deg, #F47521, #FF9A3D);
        color: white;
    }
    
    /* Watch Now Section */
    .watch-now-section {
        background: linear-gradient(135deg, 
            rgba(30, 30, 50, 0.9), 
            rgba(40, 40, 60, 0.8));
        border-radius: 18px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .watch-now-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: radar 8s linear infinite;
    }
    
    @keyframes radar {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Sidebar Enhancements */
    .css-1d391kg, .css-1lcbmhc {
        background: rgba(12, 12, 12, 0.9) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(20, 20, 35, 0.7) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 107, 107, 0.5);
        box-shadow: 0 15px 30px rgba(255, 107, 107, 0.2);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FF6B6B, #FFD93D);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FFD93D, #6BCF7F);
    }
    
    /* Analytics Specific Styles */
    .analytics-hero {
        background: linear-gradient(135deg, 
            rgba(12, 12, 12, 0.95) 0%, 
            rgba(26, 26, 46, 0.9) 50%, 
            rgba(15, 52, 96, 0.85) 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.5),
            0 0 100px rgba(77, 150, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .analytics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .analytics-card {
        background: linear-gradient(135deg, 
            rgba(30, 30, 45, 0.9), 
            rgba(20, 20, 35, 0.9));
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .analytics-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FF6B6B, #FFD93D, #6BCF7F);
        background-size: 200% 100%;
        animation: loadingBar 3s ease-in-out infinite;
    }
    
    .analytics-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(255, 107, 107, 0.2);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FF6B6B, #FFD93D, #6BCF7F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 10px 0;
    }
    
    .stat-label {
        text-align: center;
        font-size: 1.1rem;
        color: #FFD93D;
        font-weight: 600;
    }
    
    .trend-indicator {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .trend-up {
        background: linear-gradient(135deg, #6BCF7F, #4CAF50);
        color: white;
    }
    
    .trend-down {
        background: linear-gradient(135deg, #FF6B6B, #E53935);
        color: white;
    }
    
    .insight-card {
        background: linear-gradient(135deg, 
            rgba(40, 40, 60, 0.8), 
            rgba(30, 30, 50, 0.9));
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #FFD93D;
    }
    
    .progress-ring {
        transform: rotate(-90deg);
    }
    
    .progress-ring-circle {
        transition: stroke-dashoffset 0.35s;
        transform: rotate(90deg);
        transform-origin: 50% 50%;
    }
    </style>
    
    <div class="particles" id="particles-js"></div>
    
    <script>
    function createParticles() {
        const particles = document.getElementById('particles-js');
        const particleCount = 30;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            
            const size = Math.random() * 2 + 1;
            const left = Math.random() * 100;
            const animationDuration = Math.random() * 6 + 4;
            const animationDelay = Math.random() * 5;
            
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${left}vw`;
            particle.style.animationDuration = `${animationDuration}s`;
            particle.style.animationDelay = `${animationDelay}s`;
            
            particles.appendChild(particle);
        }
    }
    
    // Initialize particles when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createParticles);
    } else {
        createParticles();
    }
    </script>
    """, unsafe_allow_html=True)

# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(
        page_title="CodeFlix", 
        page_icon="🎬", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    # Initialize everything
    inject_color_bends_background()  # Add ColorBends background first
    inject_advanced_style()          # Then add enhanced CSS
    init_db()
    init_session_state()
    
    # Check environment first
    if not check_environment():
        st.stop()
    
    # Check and seed database if needed
    check_and_seed_database()
    
    # Validate API key
    omdb = RateLimitedOMDbAPI()
    if not omdb.validate_api_key():
        st.error("⚠️ Movie database API key is invalid or not working. Some features may be limited.")
    
    movies_data = get_movies_safe()
    stats = get_stats()
    
    # -----------------------------
    # Enhanced Sidebar with Streaming
    # -----------------------------
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <h1 style='color: #FF6B6B; font-size: 2.5rem; font-weight: 900; margin: 0; font-family: "Orbitron", sans-serif;'>CODEFLIX</h1>
            <p style='color: #FFD93D; font-size: 1rem; margin: 0; font-family: "Rajdhani", sans-serif;'>Your Personal Movie Universe</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Navigation with Streaming and Anime
        page = st.radio("NAVIGATION", [
            "🏠 DASHBOARD", 
            "➕ ADD MOVIES", 
            "🎬 MY COLLECTION", 
            "🎯 WATCH NOW",
            "🔍 AI FINDER",
            "📊 ANALYTICS"
        ], index=0)
        
        st.markdown("### 📈 LIVE STATS")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", stats['total_movies'])
        with col2:
            st.metric("Watched", stats['watched_count'])
        
        st.markdown(f"**Completion:** {stats['completion_rate']:.1f}%")
        if stats['average_rating'] > 0:
            st.markdown(f"**Avg Rating:** {stats['average_rating']:.1f} ⭐")
        if stats['in_theaters_count'] > 0:
            st.markdown(f"**In Theaters:** {stats['in_theaters_count']} 🎟️")

    # -----------------------------
    # Page Routing
    # -----------------------------
    if page == "🏠 DASHBOARD":
        show_dashboard(movies_data, stats)
    elif page == "➕ ADD MOVIES":
        show_add_movies_page()
    elif page == "🎬 MY COLLECTION":
        show_collection_page(movies_data)
    elif page == "🎯 WATCH NOW":
        show_streaming_page(movies_data)
    elif page == "🔍 AI FINDER":
        show_enhanced_ai_finder_page()
    elif page == "📊 ANALYTICS":
        show_analytics_page(movies_data)

if __name__ == "__main__":
    main()
