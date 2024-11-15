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
7. Launch Miniconda or Miniforge Prompt. Create and activate an environment as,
```
   > conda create -n yolov5-4.0 python=3.8 -c conda-forge
   > conda activate yolov5-4.0
```
8. Go to the yoneoLocr directory and install python libraries as,
```
   > conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch -c conda-forge
   > pip install -r requirements.txt
```
9. Put shortcuts, yoneoHole, yoneoXtal, yoneoDiff, and yoneoLowmagXtal on the desktop.
10. Launch yoneoLocrWatch.py from the shortcuts.
11. If the windows disappear immediately, try the following commands.
```
   > conda activate yolov5-4.0
   > pip uninstall Pillow
   > pip install Pillow
```
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
