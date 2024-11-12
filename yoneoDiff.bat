call C:\miniforge3\Scripts\activate.bat yolov5-4.0
C:
cd \ProgramData\yoneoLocr\
python .\yoneoLocrWatch-yolov5.py --weights weights\diffxall1024_210307.pt --img-size 1024 --object diff --diff-good --diff-soso --diff-bad

REM --diffgood : Choose "good" even if the socre is lower
REM --diffsoso : Choose "soso" even if the socre is lower
REM --diffbad  : Choose "bad" even if the socre is lower
REM 1. diffgoodx1024_210304.pt # for protein and organic semi-conductor crystals
REM 2. diffsosox1024_210305.pt # for other crystals including ice
REM 3. diffgoodicex1024_210307.pt # for 1. + ice
REM 4. diffxall1024_210307.pt # for 1 + 2
