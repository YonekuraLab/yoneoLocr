set exepath="C:\ProgramData\yoneoLocr"
set watchdir="WatchLowmagXtal"
for %%i in (%1*.jpg) do (
copy %%i %exepath%
echo %%~ni.jpg %2 > %exepath%\%watchdir%\InputImage.txt
timeout /t 2
move %exepath%\%%~ni.log %%~di%%~pi%%~ni.mrc.log
del  %exepath%\%%~ni.jpg
)




