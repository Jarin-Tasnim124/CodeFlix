# CodeFlix — Your Personal Movie Universe

CodeFlix is an interactive, Streamlit-based web application for exploring and tracking your favorite movies and anime. It features an **Enhanced AI Finder** powered by Gemini, allowing you to ask for recommendations by mood, plot, or actors, with explainable reasoning and dynamic feedback loops.

---

## 🚀 Prerequisites

Before you begin, ensure you have the following installed:
1. **Python 3.10 to 3.12** — [Download Python here](https://www.python.org/downloads/). *Note: Ensure you check "Add Python to PATH" during installation on Windows.*
2. **Git** (Optional, to clone the repo) — [Download Git here](https://git-scm.com/downloads).
3. **VS Code** (Optional but recommended) — [Download VS Code](https://code.visualstudio.com/).

---

## 🛠️ Step-by-Step Installation

### 1. Extract and Open the Project
- Download the CodeFlix zip file and extract it to your preferred location.
- *Tip: If there is an existing `.venv` folder from a previous installation, it is highly recommended to delete it to ensure a clean start.*
- Open the extracted folder in VS Code or your terminal.

### 2. Create and Activate a Virtual Environment

Open a new terminal inside the project directory and run the following commands:

**For Windows:**
```powershell
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**For macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
With the virtual environment active, install the required packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Configuration (.streamlit/secrets.toml)

CodeFlix relies on external APIs for complete functionality, especially the AI Finder component. You need to configure a secrets file.

1. Create a `.streamlit` folder in the root directory (if it doesn't exist).
2. Inside `.streamlit`, create a file named `secrets.toml`.
3. Add your API keys to the file:

```toml
# Get a free API key at http://www.omdbapi.com/apikey.aspx
OMDB_API_KEY = "your_omdb_api_key_here"

# Get a free Gemini API key at https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "your_gemini_api_key_here"

DATABASE_URL = "movies.db"
DEBUG = "false"
```
*(If you run without these keys, the app will fall back to local sample data and simplified recommendations).*

---

## 🏃 Running the Application

1. **Seed the database (Optional but recommended for first run):**
   ```bash
   python seed.py
   ```

2. **Start the Streamlit Server:**
   ```bash
   streamlit run app.py
   ```

3. The application should open automatically in your browser at `http://localhost:8501`.

Enjoy exploring your movie universe! 🎬🍿
