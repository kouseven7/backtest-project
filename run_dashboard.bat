@echo off
cd /d "C:\Users\imega\Documents\my_backtest_project\logs"
start "" "http://localhost:8765/paper_trade/dashboard.html"
"C:\Users\imega\Documents\my_backtest_project\.venv-3\Scripts\python.exe" -m http.server 8765
pause
