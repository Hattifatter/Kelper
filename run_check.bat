@echo off
cd /d %~dp0
call .venv\Scripts\activate
python src\check_environment.py
pause
