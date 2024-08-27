@echo off
set PORT=9099
set HOST=127.0.0.1

uvicorn main:app --host %HOST% --port %PORT% --forwarded-allow-ips '*'