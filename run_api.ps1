# PowerShell: run dev server
$env:PYTHONPATH = ".;src"
uvicorn api.app.main:app --reload --port 8080
