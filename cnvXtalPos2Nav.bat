call C:\Anaconda3\Scripts\activate.bat yolov5-4.0
REM call C:\ProgramData\Miniconda3\Scripts\activate.bat yolov5-4.0

set exepath="C:\ProgramData\yoneoLocr-yolov5"
python %exepath%\cnvxtalpos2Nav.py %1 -m %2 -o %3 -c %4


