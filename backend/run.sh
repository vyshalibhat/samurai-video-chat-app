
#!/bin/bash
cd "$(dirname "$0")"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
