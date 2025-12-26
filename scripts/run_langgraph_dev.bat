@echo off
REM Script to run langgraph dev with LANGSMITH_API_KEY
REM Reads key from root .env file and sets as environment variable

setlocal enabledelayedexpansion

REM Path to root .env
set ROOT_ENV=%~dp0..\.env
set LANGGRAPH_DIR=%~dp0..\src\langgraph_server

REM Check root .env
if not exist "%ROOT_ENV%" (
    echo File .env not found in project root
    echo Create .env file with LANGSMITH_API_KEY
    exit /b 1
)

REM Read LANGSMITH_API_KEY from root .env
set LANGSMITH_API_KEY=
for /f "usebackq tokens=*" %%a in ("%ROOT_ENV%") do (
    set "line=%%a"
    set "line=!line: =!"
    if "!line!" neq "" (
        for /f "tokens=1* delims==" %%b in ("!line!") do (
            if "%%b"=="LANGSMITH_API_KEY" (
                set "LANGSMITH_API_KEY=%%c"
                set "LANGSMITH_API_KEY=!LANGSMITH_API_KEY:"=!"
                set "LANGSMITH_API_KEY=!LANGSMITH_API_KEY:'=!"
            )
        )
    )
)

REM If not found, check langgraph_server/.env
if "!LANGSMITH_API_KEY!"=="" (
    set LANGGRAPH_ENV=%LANGGRAPH_DIR%\.env
    if exist "%LANGGRAPH_ENV%" (
        for /f "usebackq tokens=*" %%a in ("%LANGGRAPH_ENV%") do (
            set "line=%%a"
            set "line=!line: =!"
            if "!line!" neq "" (
                for /f "tokens=1* delims==" %%b in ("!line!") do (
                    if "%%b"=="LANGSMITH_API_KEY" (
                        set "LANGSMITH_API_KEY=%%c"
                        set "LANGSMITH_API_KEY=!LANGSMITH_API_KEY:"=!"
                        set "LANGSMITH_API_KEY=!LANGSMITH_API_KEY:'=!"
                    )
                )
            )
        )
    )
)

REM Set environment variable
if not "!LANGSMITH_API_KEY!"=="" (
    set "LANGSMITH_API_KEY=!LANGSMITH_API_KEY: =!"
    echo LANGSMITH_API_KEY set
) else (
    echo LANGSMITH_API_KEY not found in .env files
    echo LangGraph will work without monitoring
)

REM Change to langgraph_server directory and run
cd /d "%LANGGRAPH_DIR%"
if errorlevel 1 (
    echo Failed to change to %LANGGRAPH_DIR%
    exit /b 1
)

echo.
echo Starting langgraph dev...
echo.

langgraph dev

endlocal

