@echo off
REM CourtMind Daily Evening Update (5pm ET)
REM 1. Refresh lineups (final OUT/injury list before games)
REM 2. Regenerate games data with latest odds
REM 3. Regenerate top 10 picks

echo [%date% %time%] Starting evening update... >> C:\Users\user\CourtMind\daily_log.txt

REM Refresh lineups (critical - final OUT list before games)
echo [%date% %time%] Refreshing lineups and OUT list (evening)... >> C:\Users\user\CourtMind\daily_log.txt
curl -X POST "https://courtmind-api.onrender.com/api/lineups/refresh" >> C:\Users\user\CourtMind\daily_log.txt 2>&1
echo. >> C:\Users\user\CourtMind\daily_log.txt

REM Wait 5 seconds
timeout /t 5 /nobreak > nul

REM Generate games data with fresh odds
echo [%date% %time%] Generating games data (evening refresh)... >> C:\Users\user\CourtMind\daily_log.txt
curl -X POST "https://courtmind-api.onrender.com/api/games/generate" >> C:\Users\user\CourtMind\daily_log.txt 2>&1
echo. >> C:\Users\user\CourtMind\daily_log.txt

REM Wait 5 seconds
timeout /t 5 /nobreak > nul

REM Generate top 10 picks with fresh data
echo [%date% %time%] Generating top 10 picks (evening refresh)... >> C:\Users\user\CourtMind\daily_log.txt
curl -X POST "https://courtmind-api.onrender.com/api/top-picks/generate?limit=10" >> C:\Users\user\CourtMind\daily_log.txt 2>&1
echo. >> C:\Users\user\CourtMind\daily_log.txt

echo [%date% %time%] Evening update complete! >> C:\Users\user\CourtMind\daily_log.txt
echo ---------------------------------------- >> C:\Users\user\CourtMind\daily_log.txt
