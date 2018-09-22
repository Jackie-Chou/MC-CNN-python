import os
import struct
import numpy as np
import cv2

def readPfm(filename):
    f = open(filename, 'r')
    line = f.readline()
    assert line.strip() == "Pf" # one sample per pixel
    line = f.readline()
    items = line.strip().split()
    width = int(items[0])
    height = int(items[1])
    line = f.readline()
    if float(line.strip()) < 0:  # little-endian
        fmt = "<f"
    else:
        fmt = ">f"
    maps = np.ndarray([height, width], dtype=np.float32)
    for h in range(height-1, -1, -1):
        for w in range(width):
            sample = f.read(4)
            maps[h, w], = struct.unpack(fmt, sample)
    f.close()
    return maps

def parseCalib(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    
    line = lines[4].strip()
    idx = line.find('=')
    width = int(line[idx+1:])

    line = lines[5].strip()
    idx = line.find('=')
    height = int(line[idx+1:])

    line = lines[6].strip()
    idx = line.find('=')
    ndisp = int(line[idx+1:])
    return height, width, ndisp

def normal(mean, std_dev):
    constant1 = 1. / (np.sqrt(2*np.pi) * std_dev)
    constant2 = -1. / (2 * std_dev * std_dev)
    return lambda x: constant1 * np.exp(constant2 * ((x - mean)**2))

def saveDisparity(disparity_map, filename):
    assert len(disparity_map.shape) == 2
    cv2.imwrite(filename, disparity_map)

def writePfm(disparity_map, filename):
    assert len(disparity_map.shape) == 2
    height, width = disparity_map.shape
    disparity_map = disparity_map.astype(np.float32)
    o = open(filename, "w")
    # header
    o.write("Pf\n")
    o.write("{} {}\n".format(width, height))
    o.write("-1.0\n")
    # raster
    # NOTE: bottom up
    # little-endian
    fmt = "<f"
    for h in range(height-1, -1, -1):
        for w in range(width):
            o.write(struct.pack(fmt, disparity_map[h, w]))
    o.close()

def saveTimeFile(times, path):
    o = open(path, "w")
    o.write("{}".format(times))
    o.close()

def testMk(dirName):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

def recurMk(path):
    items = path.split("/")
    prefix = "/"
    for item in items:
        prefix = os.path.join(prefix, item)
        testMk(prefix)

