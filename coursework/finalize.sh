#!/bin/bash
# Двухпроходная сборка: 1) docx -> pdf, 2) номера страниц заголовков -> содержание, 3) финальный docx/pdf
set -e
cd "$(dirname "$0")"
python3 build_coursework.py
soffice --headless --convert-to pdf WeldSight_kursovaya.docx >/dev/null 2>&1
python3 find_pages.py
python3 build_coursework.py toc.json
soffice --headless --convert-to pdf WeldSight_kursovaya.docx >/dev/null 2>&1
python3 find_pages.py check
