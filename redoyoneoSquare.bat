set exepath="C:\ProgramData\yoneoLocr"
set watchdir="WatchSquare"
if "%2"=="" ( set wtime="3" ) else (set wtime="%2" )
for %%i in (%1*.jpg) do (
echo %%i > %exepath%\%watchdir%\InputImage.txt
c:\Windows\System32\timeout /t %wtime%
move %exepath%\%%~ni.sq.log %%~di%%~pi%%~ni.sq.log
)




