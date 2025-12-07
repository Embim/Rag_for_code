@echo off
echo Creating symlinks for repositories...

:: Удалить старые симлинки если есть
if exist "data\repos\***" rmdir "data\repos\***"
if exist "data\repos\***" rmdir "data\repos\***"

:: Создать новые симлинки
mklink /D "data\repos\***" "F:\***"
mklink /D "data\repos\***" "F:\***o"

echo.
echo Done! Symlinks created:
echo   data/repos/*** -> F:/***
echo   data/repos/*** -> F:/***
echo.

pause
