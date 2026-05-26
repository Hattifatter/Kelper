@echo off
cd /d %~dp0
call .venv\Scripts\activate
python src\convert_kepler_fits_to_parquet.py
pause
