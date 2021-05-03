set exepath="C:\ProgramData\yoneoLocr"
set watchdir="WatchLowmagXtal"
for %%i in (%1*.jpg) do (
echo %%i %2 > %exepath%\%watchdir%\InputImage.txt
c:\Windows\System32\timeout /t 1
move %exepath%\%%~ni.log %%~di%%~pi%%~ni.mrc.log
)




