@echo off
REM CourtMind Daily Update - Complete Workflow
REM Run this once per day to update all predictions with Odds API data
REM
REM This script:
REM   1. Refreshes lineups from Rotowire
REM   2. Fetches player props from Odds API (DK + FD)
REM   3. Fetches game odds from Odds API
REM   4. Generates games predictions
REM   5. Generates TOP 10 picks with calibrated edge requirements
REM
REM Output is logged to daily_update_log.txt

echo ================================================================
echo CourtMind Daily Update
echo ================================================================
echo Started: %date% %time%
echo.

python "C:\Users\user\CourtMind\run_daily_update.py" >> "C:\Users\user\CourtMind\daily_update_log.txt" 2>&1

echo.
echo Completed: %date% %time%
echo ================================================================
