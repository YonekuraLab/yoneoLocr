![Top](yoneo.ico)
# yoneoLocr
### Real-time object locator for cryo-EM data collection
#### --- You only navigate EM once ---
210202 Koji Yonekura, RIKEN SPring-8 / Tohoku University<BR>
&nbsp;&nbsp;&nbsp;Derived from detect.py in yolov5<BR>
210403 Version 1.0<BR>
241207 Instructions for CUDA 11<BR>
&nbsp;&nbsp;Note: Windows 10 N and 11 N require Media Feature Pack (Media Player) for OpenCV (import cv2) to work properly.
### Reference
* https://biorxiv.org/cgi/content/short/2021.04.07.438905v1
* https://www.nature.com/articles/s42003-021-02577-1
### Installation
1. Download yoneoLocr-main.zip from https://github.com/YonekuraLab/yoneoLocr.
2.	Extract the zip file and put the whole directory as yoneoLocr in C:\ProgramData\ of a camera control Windows PC.
3.	Set the property of batch files to “full control” from the Security tab if needed.
4.	Install ImageMagick.
5.	Install CUDA 10 or 11 (see below).
6.	Go to the yoneoLocr directory and install other python modules as,
```
   > pip install -r requirements.txt
```
7. Put shortcuts, yoneoHole, yoneoXtal, yoneoDiff, and yoneoLowmagXtal on the desktop.
8. Launch yoneoLocrWatch.py from the shortcuts.
9. If the windows disappear immediately, try the following commands.
```
   > conda activate yolov5-4.0
   > pip uninstall Pillow
   > pip install Pillow
```
#### CUDA 10
1. Install CUDA Toolkit 10.1 and cuDNN 10.1 for a K3 control PC with Windows Server 2012R2.
2. Install Microsoft Build Tools for Visual Studio (vs_buildtools) if needed.
3. Launch Miniconda or Miniforge Prompt. Create and activate an environment as,
```
   > conda create -n yolov5-4.0 python=3.8 -c conda-forge
   > conda activate yolov5-4.0
```
4. Install PyTorch as,
```
   > conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch -c conda-forge
```
&nbsp;&nbsp;&nbsp;or,
```
   > pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
#### CUDA 11
1. Install CUDA Toolkit 11.8 for Windows 10 and RTX-3090 or newer.
2. Launch Miniconda or Miniforge Prompt. Create and activate an environment as,
```
   > conda create -n yolov5-4.0 python=3.11 -c conda-forge
   > conda activate yolov5-4.0
```
3. Install PyTorch as,
```
   > conda install pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
4. Copy models/experimentalCUDA11.py to models/experimental.py and models/yoloCUDA11.py to models/yolo.py

### Command line options
* Select running mode.
 ```
   --object hole / xtal / diff / lowmagxtal
```
* A confidence threshold for object selection in hole and lowmagxtal modes. Default 0.4.
```
   --conf-sel 0.4
```
* Delete output file showing objects enclosed with boxes. Default: no.
```
   --delout yes / no
```
* Include ice crystals for positioning in xtal mode. Default: no.
```
   --ice yes / no
```
* Other options in the original script detect.py in YOLOv5 are also available.
### Notes
* A weight file is included only for "hole" due to limitation of the file size at the github site. Other weights are available from our web site.
