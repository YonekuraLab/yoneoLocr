call C:\miniforge3\Scripts\activate.bat yolov5-4.0
REM call C:\ProgramData\Miniconda3\Scripts\activate.bat yolov5-4.0

C:
cd \ProgramData\yoneoLocr\
python .\yoneoLocrWatch-yolov5.py --weights weights\squarex800_210829.pt --object square --conf-thres 0.1



