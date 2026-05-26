# Aerospace Data Stand

Экспериментальный стенд для обработки и визуализации данных NASA Kepler.

## Структура

- `data/raw/kepler_real/small` — малый набор FITS-файлов.
- `data/raw/kepler_real/medium` — средний набор FITS-файлов.
- `data/raw/kepler_real/large` — большой набор FITS-файлов.
- `data/samples` — Parquet-файлы после конвертации.
- `src` — исходный код проекта.
- `results` — метрики, графики, логи, скриншоты.

## Базовый запуск

```powershell
cd C:\Kelper
.\.venv\Scripts\activate
python src\check_environment.py
python src\convert_kepler_fits_to_parquet.py
```
