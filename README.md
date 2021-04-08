# yoneoLocr
### Real-time object locator for cryo-EM data collection
#### --- You only navigate EM once ---
210202 Koji Yonekura, RIKEN SPring-8 / Tohoku University<BR>
&nbsp;&nbsp;&nbsp;Derived from detect.py in yolov5<BR>
&nbsp;&nbsp;&nbsp;cnvxtalpos2Nav.py: Derived from convLM2DIFF.py (by Kiyofumi Takaba)<BR>
210403 Version 1.0<BR>
### Installation
1. Put the yoneoLocr directory on C:\ProgramData\ in a camera control Windows PC.
2. Install CUDA toolkit 10.1 and cuDNN 10.1 for a K3 control PC if the operating system of the PC is Windows Server 12R.  CUDA 10.1 is the newest version supporting Windows Server 12R. Newer versions of CUDA and cuDNN are available for Windows 10.
3. Install Microsoft Build Tools for Visual Studio (vs_buildtools) if needed.
4. Install ImageMagick.
5. Launch Anaconda Prompt. Make and activate an Anaconda environment as,
```
   > conda create -n yolov5-4.0 python=3.8
   > conda activate yolov5-4.0
```
6. Go to the yoneoLocr directory and install python libraries as,
```
   > conda install -c pytorch torchvision cudatoolkit=10.1 
   > pip install -r requirements.txt
```
7. Make shortcuts of yoneoLocrHole.bat, yoneoLocrXtal.bat, yoneoLocrDiff.bat, and yoneoLocrLowmagXtal.bat on the desktop.
8. Launch yoneoLocrWatch.py from the shortcuts.
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
* The github site includes weights for only "hole" and "lowmagxtal" due to file size limit. Other weights are downloadable from our web site.
