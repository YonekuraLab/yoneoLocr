# Convert xtal positions in low mag images determined by yoneoLocrWatch
#         to SerialEM nav file
# 210315 K. Yonekura
#        Derived from convLM2DIFF.py by K. Takaba

import argparse
import os, glob, math
from pathlib import Path
from natsort import natsorted

parser = argparse.ArgumentParser(description='Make xtal position.')

parser.add_argument(
    'input_path_nav',
    default=None,
    help='path of *.nav file of serialEM'
)

parser.add_argument(
    '-m',
    '--mdoc',
    type=str,
    default='*.mdoc',
    help='output filename (*.mdoc)'
)

parser.add_argument(
    '-o',
    '--output',
    type=str,
    default='_new.nav',
    help='output filename (_new.nav)'
)

parser.add_argument(
    '-c',
    '--cutlevel',
    type=int,
    default=1,
    help='including thick crystals (1) and those at edges (2)'
)

def readtag(tag_data, key):
    if not key in tag_data:
        return 0
    if 'Item' in key:
        return tag_data['[Item'].split(']')[0]
    else:
        return tag_data[key].split('\n')[0]

def nav2tags(infile):
    tags = []
    tag = {}
    with open(infile) as f:
        for line in f:
            if line == '\n':
                if tag:
                    tags.append(tag)
                tag = {}
            if line.split(' = ')[0] != '\n':
                tag[line.split(' = ')[0]] = line.split(' = ')[1]
        if tag:
            if tag['[Item'] != tags[-1]['[Item']:
                tags.append(tag)
    return tags

def readmdoc(infile):
    with open(infile) as f:
        for line in f:
            word = line.split()
            if len(word) < 3:
                continue
            if word[0] == "PixelSpacing":
                pixsize1 = eval(word[2])
            elif word[0] == "ImageSize":
                xdim1, ydim1 = eval(word[2]),eval(word[3])
            elif word[0] == "StagePosition":
                xs1, ys1 = eval(word[2]), eval(word[3])
            elif word[0] == "StageZ":
                zs1 = eval(word[2])
            elif word[0] == "RotationAngle":
                rotangle1 = eval(word[2])
    return pixsize1, xdim1, ydim1, xs1, ys1, zs1, rotangle1 

def readposlogs(infile, psarry):
    i = 0
    with open(infile) as f:
        for line in f:
            word = line.split()
            if len(word) != 6:
                continue
            cls = eval(word[1])
            if cls  <= cutlevel:
                x1, y1 = eval(word[2]), eval(word[3])
                # dist = (x1**2 + y1**2)**0.5
                postmp = [ x1, y1, eval(word[4]), eval(word[5]), \
                           eval(word[0]), cls ]
                psarry.append(postmp)
                i += 1
        print ("Read {:d} positions from {:}".format(i, infile))
    return i

def outnav(f1, xyz, pixsize1, xdim1, ydim1, xs1, ys1, zs1, rot1, \
           base1, count1, inm1):
    #if readtag(tags_read[i], 'StageXYZ') == 0:
    #    print('Item {:} is ignored.'.format(i))
    #    continue
    f1.write("[Item = {:}-{:}]\n".format(base1, count1))
    #f1.write("Color = {:}\n".format(readtag(tags_read[-1], 'Color')))
    f1.write("Color = 3\n")
    xor = xyz[0] - 0.5
    yor = xyz[1] - 0.5
    rot =  math.radians(rot1)
    xor1 = math.cos(rot)*xor - math.sin(rot)*yor
    yor1 = math.sin(rot)*xor + math.cos(rot)*yor
    newx = (xor1 * pixsize1 * xdim1)/10000. + xs1
    # convert to Angstrom to um
    newy = (yor1 * pixsize1 * ydim1)/10000. + ys1
    f1.write("StageXYZ = {:.3f} {:.3f} {:.3f}\n".format(newx, newy, zs1))
    #f1.write("NumPts = {:}\n".format(readtag(tags_read[-1], 'NumPts')))
    #f1.write("Regis = {:}\n".format(readtag(tags_read[-1], 'Regis')))
    #f1.write("Type = {:}\n".format(readtag(tags_read[-1], 'Type')))
    f1.write("NumPts = 1\n")
    f1.write("Regis = 1\n")
    f1.write("Type = 0\n")
    f1.write("Note = yoneoLocr {:.3f} {:.3f} conf {:.3f} cl {:.0f} {:}\n".\
             format(xyz[0], xyz[1], xyz[4], xyz[5], inm1))
    #f1.write("RawStageXY = {:.3f} {:.3f}\n".format(newx, newy))
    f1.write("Acquire = 0\n") # 1 for 'A'
    f1.write("GroupID = {:}\n".format("10000"+base1))
    #f1.write("GroupID = 123456789\n")
    f1.write("PtsX = {:.3f}\n".format(newx))
    f1.write("PtsY = {:.3f}\n\n".format(newy))
   
if __name__ == "__main__":
    args = parser.parse_args()
    
    print(" Input file: {:}".format(args.input_path_nav))
    tags_read = nav2tags(args.input_path_nav)
    #item_lastid = int(readtag(tags_read[-1], '[Item').split("-")[0])
        
    mdocbase  = args.mdoc
    print(" mdoc files: {:}*.mdoc".format(mdocbase))
    mdocs   = glob.glob(mdocbase+'*.mdoc')
    outfile = args.output
    print("Output file: {:}".format(outfile))
    cutlevel = args.cutlevel
    if cutlevel == 0:
        print(" Exclude thick crystals and those at edge\n")
    elif cutlevel == 1:
        print(" Exclude crystals at edge\n")
    elif cutlevel == 1:
        print(" Include thick crystals and those at edge\n")
    fout = open(outfile, 'w')
    fout.write('AdocVersion = 2.00\n')
    fout.write('LastSavedAs = {:}\n\n'.format(outfile))
    for mfile in natsorted(mdocs):
        pixsize, xdim, ydim, xs, ys, zs, rotangle = readmdoc(mfile)
        p0 = Path(mfile)
        baseid = p0.stem.split("_")[1].split(".")[0]
        poslog = str(p0.parent) + "\\" + p0.stem + ".log"
        if not os.path.exists(poslog) :
            print ("File \"%s\" not found" % poslog)
        else :
            psarray = []
            pcounts = readposlogs(poslog, psarray)
            psarray = sorted(psarray, key=lambda q: q[1]+q[0])
            i = 1
            for xyzconf in psarray:
                count = i
                outnav(fout, xyzconf, pixsize, xdim, ydim, xs, ys, zs,
                       rotangle, baseid, count, p0.stem+".log")
                i += 1
    fout.close()
    




