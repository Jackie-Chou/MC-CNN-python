// disp2pfm.cpp 

// converts pgm/png disparities to pfm, optionally dividing by dispfact (use 2 for half-size, etc.)
// if mapzero=1, 0's are considered unknown disparities and mapped to infinity

// DS 10/3/2013

static const char *usage = "\n  usage: %s disp.pgm disp.pfm [dispfact=1 [mapzero=0]]\n";

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "imageLib.h"

int verbose = 1;

// convert gray disparity image into a float image
// scale values by dispfact and convert 0's (unk values) into inf's 
void disp2floatImg(CByteImage img, CFloatImage &fimg, int dispfact, int mapzero)
{
    CShape sh = img.Shape();
    int width = sh.width, height = sh.height;
    sh.nBands = 1; 
    fimg.ReAllocate(sh);

    float s = 1.0 / dispfact;
    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	    int v = img.Pixel(x, y, 0);
	    float f = s * v;
	    if (v == 0 && mapzero)
		f = INFINITY;
	    fimg.Pixel(x, y, 0) = f;
	}
    }
}

int main(int argc, char *argv[])
{
    try {
	int requiredargs = 2;
	int optionalargs = 2;
	if (argc >= requiredargs + 1 && argc <= requiredargs + optionalargs + 1) {
	    int argn = 1;
	    char *dispname = argv[argn++];
	    char *fdispname = argv[argn++];
	    int dispfact = 1;
	    if (argc > argn)
		dispfact = atoi(argv[argn++]);
	    int mapzero = 0;
	    if (argc > argn)
		mapzero = atoi(argv[argn++]);
      
	    CByteImage disp;
	    CFloatImage fdisp;
	    ReadImageVerb(disp, dispname, verbose);
	    disp2floatImg(disp, fdisp, dispfact, mapzero);
	    WriteImageVerb(fdisp, fdispname, verbose);
	} else
	    throw CError(usage, argv[0]);
    }
    catch (CError &err) {
	fprintf(stderr, err.message);
	fprintf(stderr, "\n");
	return -1;
    }
  
    return 0;
}
