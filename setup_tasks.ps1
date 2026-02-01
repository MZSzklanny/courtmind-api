# CourtMind - Setup Scheduled Tasks
# Run this script as Administrator to create the daily tasks

# Morning task at 6:00 AM
$morningAction = New-ScheduledTaskAction -Execute "C:\Users\user\CourtMind\daily_morning_6am.bat"
$morningTrigger = New-ScheduledTaskTrigger -Daily -At 6:00AM
Register-ScheduledTask -TaskName "CourtMind_Morning_6AM" -Action $morningAction -Trigger $morningTrigger -Description "Grade yesterday and generate top 10 picks" -Force

# Evening task at 5:00 PM
$eveningAction = New-ScheduledTaskAction -Execute "C:\Users\user\CourtMind\daily_evening_5pm.bat"
$eveningTrigger = New-ScheduledTaskTrigger -Daily -At 5:00PM
Register-ScheduledTask -TaskName "CourtMind_Evening_5PM" -Action $eveningAction -Trigger $eveningTrigger -Description "Refresh top 10 picks with latest odds" -Force

Write-Host "Tasks created successfully!"
Write-Host ""
Write-Host "To verify, run: schtasks /query /tn CourtMind*"
