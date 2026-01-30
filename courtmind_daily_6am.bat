@echo off
REM CourtMind Daily Update - Runs at 6:00 AM
REM Updates: game data, injuries, parquet files

cd /d C:\Users\user\CourtMind
echo [%date% %time%] CourtMind Daily Update Starting >> daily_update_log.txt

python courtmind_daily_update.py

echo [%date% %time%] CourtMind Daily Update Finished >> daily_update_log.txt
