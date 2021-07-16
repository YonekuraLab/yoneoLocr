![Top](yoneo.ico)
# yoneoLocr
### Real-time object locator for cryo-EM data collection
#### --- You only navigate EM once ---
210202 Koji Yonekura, RIKEN SPring-8 / Tohoku University<BR>
&nbsp;&nbsp;&nbsp;Derived from detect.py in yolov5<BR>
&nbsp;&nbsp;&nbsp;cnvxtalpos2Nav.py: Derived from convLM2DIFF.py (by Kiyofumi Takaba)<BR>
210403 Version 1.0<BR>
### Reference
* https://biorxiv.org/cgi/content/short/2021.04.07.438905v1
### Installation
1. Download yoneoLocr-main.zip from https://github.com/YonekuraLab/yoneoLocr.
2.	Extract the zip file and put the whole directory as yoneoLocr in C:\ProgramData\ of a camera control Windows PC.
3.	Set the property of batch files to “full control” from the Security tab if needed.
4.	Install CUDA Toolkit 10.1 and cuDNN 10.1 for a K3 control PC if the operating system of the PC is Windows Server 2012R2. Newer versions of CUDA and cuDNN are available for Windows 10.
5. Install Microsoft Build Tools for Visual Studio (vs_buildtools) if needed.
6. Install ImageMagick.
7. Launch Anaconda Prompt. Make and activate an Anaconda environment as,
```
   > conda create -n yolov5 python=3.8
   > conda activate yolov5
```
8. Go to the yoneoLocr directory and install python libraries as,
```
   > conda install -c pytorch torchvision cudatoolkit=10.1 
   > pip install -r requirements.txt
```
9. Put shortcuts, yoneoHole, yoneoXtal, yoneoDiff, and yoneoLowmagXtal on the desktop.
10. Launch yoneoLocrWatch.py from the shortcuts.
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
* Equalize image histogram. Only effective in hole mode. Default: yes.
```
   --equalize yes / no
```
* Other options in the original script detect.py in YOLOv5 are also available.
### Notes
* The github site includes a weight for only "hole" due to file size limit. Other weights are downloadable from our web site.
