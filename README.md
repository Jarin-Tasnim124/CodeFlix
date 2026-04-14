# CodeFlix — Streamlit Movie Watchlist

## How to run and create suitable environment

1.Download the Codeflix Zip file and extract it.

2.Delete any Previous Venv. folder(virtual Environment folder) if initially seen in the main file.

2.Then open the folder in Vscode.

3.Install Python 3.12 on the device with run pathway while installing(https://www.python.org/downloads/release/python-3120/)

4. Then run the following commands chronologically opening new Terminal: 

1. py -3.12 -m venv .venv
2. Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
3. .\.venv\Scripts\Activate.ps1
4. pip install streamlit
5. python.exe -m pip install --upgrade pip
6. pip install requests
7. pip install plotly
8. pip install matplotlib
9. python seed.py
10. streamlit run app.py


### Optional (OMDb for AI Finder)
Set an environment variable `OMDB_API_KEY` for better fallback results and plots.

**macOS/Linux:**
```bash
export OMDB_API_KEY=your_key_here
```
**Windows (PowerShell):**
```powershell
setx OMDB_API_KEY "your_key_here"
```

Enjoy!
