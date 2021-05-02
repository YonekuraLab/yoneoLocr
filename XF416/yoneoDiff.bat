call C:\Anaconda3\Scripts\activate.bat yolov5-4.0
REM call C:\ProgramData\Miniconda3\Scripts\activate.bat yolov5-4.0

python .\yoneoLocrWatch-yolov5.py --weights weights\diffxall1024_210307.pt --img-size 1024 --object diff -device 0

REM 1. diffgoodx1024_210304.pt # for protein and organic semi-conductor crystals
REM 2. diffsosox1024_210305.pt # for other crystals including ice
REM 3. diffgoodicex1024_210307.pt # for 1. + ice
REM 4. diffxall1024_210307.pt # for 1 + 2
