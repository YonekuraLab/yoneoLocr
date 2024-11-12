set exepath="C:\ProgramData\yoneoLocr"
set watchdir="WatchLowmagXtal"
if "%3"=="" ( set wtime="3" ) else (set wtime="%3" )
for %%i in (%1*.jpg) do (
copy %%~di%%~pi%%~ni.sq.log %exepath%\
echo %%i %2 > %exepath%\%watchdir%\InputImage.txt
c:\Windows\System32\timeout /t %wtime%
move %exepath%\%%~ni.log %%~di%%~pi%%~ni.mrc.log
del %exepath%\%%~ni.sq.log
)




