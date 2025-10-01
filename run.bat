@echo off
setlocal
if not exist .venv\Scripts\activate (
  echo Virtual environment not found. Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python yt_comments_tool.py %*
