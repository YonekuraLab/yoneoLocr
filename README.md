# yoneoLocr
Real-time object locator for cryo-EM data collection

 210202 Koji Yonekura, RIKEN SPring-8/Tohoku University
 
                   Derived from detect.py in yolov5
 
 210403 Version 1.0

Installation.
1. Put yoneoLocr-yolov5 on C:\ProgramData\ in a camera control Windows PC.
2. Install CUDA toolkit 10.1 and cuDNN 10.1 for a K3 control PC if the operating system of the PC is Windows Server 12R.  CUDA 10.1 is the newest version supporting Windows Server 12R. Newer versions of CUDA and cuDNN are available for Windows 10.
3. Install Microsoft Build Tools for Visual Studio (vs_buildtools) if needed.
4. Install ImageMagick.
5. Launch Anaconda Prompt. Make and activate an Anaconda environment as,
 conda create -n yolov5-4.0 python=3.8
 condo activate yolov5-4.0
6. Go to the yoneoLocr-yolov5 directory and install python libraries as, 
 conda install -c pytorch torchvision cudatoolkit=10.1
 pip install -r requirements.txt
All required libraries are written in requirements.txt.
7. Make shortcuts of yoneoLocrHole.bat, yoneoLocrXtal.bat, yoneoLocrDiff.bat, and yoneoLocrLowmagXtal.bat on the desktop.
8. Launch yoneoLocrWatch.py from the shortcuts.
