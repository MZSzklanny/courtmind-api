@echo off
REM CourtMind Daily Morning Update (6am ET)
REM 1. Grade yesterday's picks
REM 2. Refresh lineups (includes OUT/injury list)
REM 3. Generate games data
REM 4. Generate top 10 picks

echo [%date% %time%] Starting morning update... >> C:\Users\user\CourtMind\daily_log.txt

REM Grade yesterday's picks
echo [%date% %time%] Grading yesterday's picks... >> C:\Users\user\CourtMind\daily_log.txt
curl -X POST "https://courtmind-api.onrender.com/api/daily-tracking/grade" >> C:\Users\user\CourtMind\daily_log.txt 2>&1
echo. >> C:\Users\user\CourtMind\daily_log.txt

REM Wait 5 seconds
timeout /t 5 /nobreak > nul

REM Refresh lineups (includes OUT players)
echo [%date% %time%] Refreshing lineups and OUT list... >> C:\Users\user\CourtMind\daily_log.txt
curl -X POST "https://courtmind-api.onrender.com/api/lineups/refresh" >> C:\Users\user\CourtMind\daily_log.txt 2>&1
echo. >> C:\Users\user\CourtMind\daily_log.txt

REM Wait 5 seconds
timeout /t 5 /nobreak > nul

REM Generate games data
echo [%date% %time%] Generating games data... >> C:\Users\user\CourtMind\daily_log.txt
curl -X POST "https://courtmind-api.onrender.com/api/games/generate" >> C:\Users\user\CourtMind\daily_log.txt 2>&1
echo. >> C:\Users\user\CourtMind\daily_log.txt

REM Wait 5 seconds
timeout /t 5 /nobreak > nul

REM Generate top 10 picks
echo [%date% %time%] Generating top 10 picks... >> C:\Users\user\CourtMind\daily_log.txt
curl -X POST "https://courtmind-api.onrender.com/api/top-picks/generate?limit=10" >> C:\Users\user\CourtMind\daily_log.txt 2>&1
echo. >> C:\Users\user\CourtMind\daily_log.txt

echo [%date% %time%] Morning update complete! >> C:\Users\user\CourtMind\daily_log.txt
echo ---------------------------------------- >> C:\Users\user\CourtMind\daily_log.txt
