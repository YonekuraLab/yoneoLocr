call C:\miniforge3\Scripts\activate.bat yolov5-4.0

conda create -n yolov5-4.0 python=3.8
conda activate yolov5-4.0
cd \ProgramData\yoneoLocr
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch 
pip install -r requirements.txt
