call C:\miniforge3\Scripts\activate.bat yolov5-4.0
REM call C:\ProgramData\Miniconda3\Scripts\activate.bat yolov5-4.0

REM python .\yoneoLocrWatch-yolov5.py --weights weights\lowmagxtalx800_210314.pt --object lowmagxtal --conf-sel 0.25

REM python .\yoneoLocrWatch-yolov5.py --weights weights\lowmagxtals800_210325.pt --object lowmagxtal --conf-sel 0.25

if "%1"=="" ( set thres="0.5" ) else (set thres="%1" )
C:
cd \ProgramData\yoneoLocr\
python .\yoneoLocrWatch-yolov5.py --weights weights\lowmagxtalx800_210505.pt --object lowmagxtal --conf-sel %thres%



