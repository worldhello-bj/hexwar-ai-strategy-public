@echo off
chcp 65001 >nul
setlocal
set HEXWAR_PORT=8080
if not "%~1"=="" set HEXWAR_PORT=%~1
echo ============================================================
echo   Hexwar 离线兵棋推演 - 本地桩服务器
echo   地址: http://127.0.0.1:%HEXWAR_PORT%/
echo   停止: Ctrl+C
echo ============================================================
start "" "http://127.0.0.1:%HEXWAR_PORT%/"
py -3.13 "%~dp0server.py"
endlocal
