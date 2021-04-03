# Locate objects such as carbon holes or crystals with yolov5 and PyTorch
# 210202 K. Yonekura, RIKEN SPring-8/Tohoku University
#          Derived from detect.py in yolov5
# 210403 Version 1.0

import argparse
import time, os, subprocess, datetime, re, math
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, \
    set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

prevdata = " "

def outxyshifts(outfile, xp1, yp1, hscore1, hclno1, inputdata1):
    with open(outfile,"w") as fdout:
        fdout.write("%f %f %f %d # %s\n" % \
                    (xp1, yp1, hscore1, hclno1, inputdata1))
        fdout.write("# %s" % datetime.datetime.today())

def outdiff(outfile, hscore1, hclno1, inputdata1):
    with open(outfile,"w") as fdout:
        fdout.write("%f %d # %s\n" % \
                    (hscore1, hclno1, inputdata1))
        fdout.write("# %s" % datetime.datetime.today())
        
def outlowmagXtalpos(outfile, xcen1, ycen1, xlen1, ylen1, score1, clno1):
    if score1 > confsel:
        with open(outfile,"a") as fdout:
            fdout.write("%f %f %f %f %f %f\n" % \
                        (score1, clno1, xcen1, ycen1, xlen1, ylen1))

def outlowmagXtalEnd(outfile, inputdata1):
    with open(outfile,"a") as fdout:
        fdout.write("# %s %s" % (inputdata1, datetime.datetime.today()))

def holemax(score1, xcen1, ycen1, xlen1, ylen1, hmax1, maxlen1,\
            xminh1, yminh1, xmaxh1, ymaxh1, hscore1, dmin1):
    hsize1 = xlen1*ylen1
    print("conf: %f hole size: %f center: %f %f" %
          (score1, hsize1, xcen1, ycen1))
    dist = math.sqrt((xcen1-0.5)*(xcen1-0.5) + (ycen1-0.5)*(ycen1-0.5))
    if hmax1*0.95 < hsize1 and dist < dmin1 and score1 > confsel :
        xminh1 = xcen1 - xlen1/2.
        if xminh1 < 0. : xminh1 = 0.
        xmaxh1 = xcen1 + xlen1/2.
        if xmaxh1 > 1. : xmaxh1 = 1.
        yminh1 = ycen1 - ylen1/2.
        if yminh1 < 0. : yminh1 = 0.
        ymaxh1 = ycen1 + ylen1/2.
        if ymaxh1 > 1. : ymaxh1 = 1.
        hscore1 = score1
        hmax1   = hsize1
        dmin1   = dist
        if maxlen1 < xlen1 and score1 > 0.5 :
            maxlen1 = xlen1
        if maxlen1 < ylen1 and score1 > 0.5 :
            maxlen1 = ylen1
    return hmax1, maxlen1, xminh1, yminh1, xmaxh1, ymaxh1, hscore1, dmin1

def xtalpos(score1, xcen1, ycen1, xlen1, ylen1, clno1, hmax1, \
            xminh1, yminh1, xmaxh1, ymaxh1, hscore1, hclno1, dmin1):
    size1 = xlen1*ylen1
    print("conf: %f size: % fcenter: %f %f" % (score1, size1, xcen1, ycen1))
    if clno1 == 1 :
        print (" This is likely ice.")
        if include_ice != 'yes':
            return hmax1,xminh1,yminh1,xmaxh1,ymaxh1,hscore1,hclno1,dmin1
    dist = math.sqrt((xcen1-0.5)*(xcen1-0.5) + (ycen1-0.5)*(ycen1-0.5))
    if dist < dmin1 and hmax1*0.5 < size1 and score1 > hscore1*0.8:
        xminh1 = xcen1 - xlen1/2.
        if xminh1 < 0. : xminh1 = 0.
        xmaxh1 = xcen1 + xlen1/2.
        if xmaxh1 > 1. : xmaxh1 = 1.
        yminh1 = ycen1 - ylen1/2.
        if yminh1 < 0. : yminh1 = 0.
        ymaxh1 = ycen1 + ylen1/2.
        if ymaxh1 > 1. : ymaxh1 = 1.
        hscore1 = score1
        hmax1   = size1        
        dmin1   = dist
        hclno1  = clno1
    return hmax1, xminh1, yminh1, xmaxh1, ymaxh1, hscore1, hclno1, dmin1

def diffchk(score1, clno1, hscore1, hclno1):
    if clno1 == 0:
        print("conf: %f : Good" % score1)
    elif clno1 == 1:
        print("conf: %f : So so" % score1)
    elif clno1 == 2:
        print("conf: %f : Bad" % score1)
    elif clno1 == 3:
        print("conf: %f : No" % score1)
    elif clno1 == 4:
        print("conf: %f : Ice" % score1)        
    if score1 > hscore1 :
        hscore1 = score1
        hclno1 = clno1
    return hscore1, hclno1

def detectloop():
    global prevdata, pcthres, newdir
    with open(watchdir + "\\InputImage.txt") as fdin:
        word = fdin.readline().split()
    inputdata = word[0]
    if prevdata == inputdata :
        prevdata = " "
        return
    prevdata = inputdata

    if not os.path.exists(inputdata) :
        print ("File \"%s\" not found" % inputdata)
        return

    binning   = 2
    lenwords = len(word)
    cthres = opt.conf_thres
    if lenwords > 2:
        binning = int(word[1])
        cthres  = float(word[2])
        print("\nIn \"%s\", binning %d, conf-thres %.3f" % \
              (inputdata, binning, cthres))
    elif lenwords > 1:
        binning   = int(word[1])
        print("\nIn \"%s\", binning %d" % (inputdata, binning))
    else :
        print("\nIn \"%s\"" % inputdata)
    if cthres != pcthres and delout != "yes" :
        newdir = "jpgout"+datetime.datetime.now().strftime("%y%m%d")+\
                 "_{:.3f}\\".format(cthres)
        if not os.path.exists(newdir) :
            os.makedirs(newdir)
    pcthres = cthres
    p0 = Path(inputdata)
    cnvimg = p0.stem + "c.jpg" 
    if binning < 8 and object_detect == 'hole':
        print("Resize by 50%% and enhance image contrast.")
        res = subprocess.call(
            "magick convert %s -resize 50%% -equalize %s" % (inputdata, cnvimg))
        source  = cnvimg
    elif ( binning <= 8 and object_detect == 'xtal') or \
         ( binning < 8  and object_detect == 'lowmagxtal'):
        print("Resize by 50%%.")
        res = subprocess.call(
            "magick convert %s -resize 50%% %s" % (inputdata, cnvimg))
        source  = cnvimg
    else :
        source  = inputdata

    save_conf = True
    save_img  = True
    dataset = LoadImages(source, img_size=imgsz)

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, cthres, opt.iou_thres,\
                                classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        print("\n ---")
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_name = newdir+p.stem + "out.jpg"
            save_log  = p0.stem + ".log"
            if os.path.exists(save_log) and object_detect == 'lowmagxtal':
                os.remove(save_log)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            # normalization gain whwh
            hmax   = 0.
            maxlen = 0.
            xshift = yshift = 0.
            xp = yp = 0.
            hscore  = 0.
            dmin  = 1.45
            xminh = yminh = xmaxh = ymaxh = 0.
            hclno = -1
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], \
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) \
                            / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    # label format
                    #with open(save_log, 'a') as f:
                    #    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #print(('%g ' * len(line)).rstrip() % line)
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, \
                                     color=colors[int(cls)], line_thickness=3)
                    score = eval(str(conf).lstrip("tensor(").split(",")[0])
                    xcen  = xywh[0]
                    ycen  = xywh[1]
                    xlen  = xywh[2]
                    ylen  = xywh[3]
                    clno  = cls
                    if object_detect == 'hole':
                        hmax,maxlen,xminh,yminh,xmaxh,ymaxh,hscore,dmin = \
                            holemax(score, xcen, ycen, xlen, ylen, hmax,
                                    maxlen, xminh, yminh, xmaxh, ymaxh,
                                    hscore, dmin)
                        hclno = clno
                    elif object_detect == 'xtal':
                        hmax,xminh,yminh,xmaxh,ymaxh,hscore,hclno,dmin = \
                            xtalpos(score, xcen, ycen, xlen, ylen, clno, hmax,
                                    xminh, yminh, xmaxh, ymaxh,
                                    hscore, hclno, dmin)
                    elif object_detect == 'diff':
                        hscore, hclno = diffchk(score, clno, hscore, hclno)
                    elif object_detect == 'lowmagxtal':
                        outlowmagXtalpos(save_log, xcen, ycen, xlen, ylen, \
                                         score, clno)
                        hclno = clno
                if object_detect == 'hole':
                    if xminh < 0.05 and (xmaxh - xminh) < maxlen:
                        xminh = (xmaxh - xminh) - maxlen
                    elif xmaxh > 0.95 and (xmaxh - xminh) < maxlen:
                        xmaxh = 1. + maxlen - (xmaxh - xminh)
                    if yminh < 0.05 and (ymaxh - yminh) < maxlen:
                        yminh = (ymaxh - yminh) - maxlen
                    elif ymaxh > 0.95 and (ymaxh - yminh) < maxlen:
                        ymaxh = 1. + maxlen - (ymaxh - yminh)
                print(" ---")
                if object_detect == 'hole' or object_detect == 'xtal' :
                    xshift = (xminh+xmaxh)/2. - 0.5
                    yshift = (yminh+ymaxh)/2. - 0.5
                    xp = -xshift
                    yp = -yshift
                    print("The stage should shift to %f %f with score %f\n" % \
                          (xp,yp,hscore)  )
                elif object_detect == 'diff':
                    if hclno == 0:
                        print("This is good with conf: %f" % hscore)
                    elif hclno == 1:
                        print("This is so so with conf: %f" % hscore)
                    elif hclno == 2:
                        print("This is bad with conf: %f" % hscore)
                    elif hclno == 3:
                        print("No xtal with conf: %f" % hscore)
                    elif hclno == 4:
                        print("This is likely ice with conf: %f" % hscore)
                                   
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_name, im0)

    print(f'Done. ({time.time() - t0:.3f}s)')
    if hclno == -1:
        print("Can't identify object")
    if object_detect == 'hole' or object_detect == 'xtal' :
        outxyshifts(save_log, xp, yp, hscore, hclno, inputdata)
    elif object_detect == 'diff':
        outdiff(save_log, hscore, hclno, inputdata)
    elif object_detect == 'lowmagxtal':
        outlowmagXtalEnd(save_log, inputdata)
                    
    if os.path.exists(cnvimg):
        os.remove(cnvimg)
    if os.path.exists(save_name) and delout == 'yes':
        os.remove(save_name)

class FileChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        detectloop()

    def on_modified(self, event):
        detectloop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights\\holes800_210222.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=800, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--conf-sel',  type=float, default=0.4, help='confidence threshold for selection of holes and xtals at a low-mag')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--object',  type=str, default='hole', help='object to be saved')
    parser.add_argument('--delout',  type=str, default='no', help='delete output (yes/no)')
    parser.add_argument('--ice',  type=str, default='no', help='include ice (yes/no)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        weights, view_img, imgsz = opt.weights, opt.view_img, opt.img_size
        # Initialize
        object_detect = opt.object
        delout = opt.delout
        include_ice  = opt.ice
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', \
                            map_location=device)['model']).to(device).eval()
        
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None
        newdir = ".\\"
        if delout != "yes" :
            newdir = "jpgout"+datetime.datetime.now().strftime("%y%m%d")+"\\"
            if not os.path.exists(newdir) :
                os.makedirs(newdir)

        event_handler = FileChangeHandler()
        confsel = opt.conf_sel
        if object_detect == "hole":
            watchdir = ".\\WatchHole"
        elif object_detect == "xtal" :
            watchdir = ".\\WatchXtal"
        elif object_detect == "diff":
            watchdir = ".\\WatchDiff"
        elif object_detect == "lowmagxtal":
            watchdir = ".\\WatchLowmagXtal"
        if not os.path.exists(watchdir) :
            os.makedirs(watchdir)

        pcthres = opt.conf_thres
        observer = Observer()
        observer.schedule(event_handler, watchdir, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            if opt.update:  # update all models (to fix SourceChangeWarning)
                strip_optimizer(opt.weights)
            observer.stop()
        observer.join()

        
