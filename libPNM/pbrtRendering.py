import sys
from PNM import *

def main():
    for i in [8, 16, 32, 64]:
        filename = '/homes/lx219/Downloads/AdvancedGraphics/simple_sphere'+str(i)+'.pfm'
        img_ = loadPFM(filename)
        max_pix = np.max(img_)
        img_ *= 255.0/max_pix
        writePPM('../Simple_sphere: {}.ppm'.format(i),img_.astype(np.uint8))

main()
