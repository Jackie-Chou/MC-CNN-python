// computemask.cpp 

// computes mask for nonoccluded regions in direction dir
// use dir = -1 for left->right, dir = 1 for right->left
// sets unknown pixels to 0, occluded pixels to 128, and nonoccluded pixels to 255

// DS 10/8/2013
// 11/22 - added dir param
// 5/12/2014 - added ydisps

static const char *usage = "\n\
  usage: %s disp0.pfm [disp0y.pfm] disp1.pfm dir mask.png [thresh=1.0]\n\
\n\
  use dir=-1 for left->right, dir=1 for right->left\n";

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "imageLib.h"

int verbose = 0;

void makeNonoccMask(CFloatImage disp0, CFloatImage disp0y, CFloatImage disp1,
		    int dir, float thresh, CByteImage &mask)
{
    CShape sh = disp0.Shape();
    int width = sh.width, height = sh.height;

    if (sh != disp1.Shape())
	throw CError("shapes differ");

    int ydisps = (sh == disp0y.Shape());

    mask.ReAllocate(sh);
    mask.ClearPixels();
    int x, y;
    for (y = 0; y < height; y++) {
	for (x = 0; x < width; x++) {
	    float dx = disp0.Pixel(x, y, 0);
	    float dy = (ydisps ? disp0y.Pixel(x, y, 0) : 0.0);
	    if (dx == INFINITY) // unknown
		continue;
	    mask.Pixel(x, y, 0) = 128; // occluded

	    // find nonocc
	    int x1 = (int)round(x + dir * dx);
	    int y1 = (int)round(y + dy);
	    if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height)
		continue;
	    float dx1 = disp1.Pixel(x1, y1, 0);

	    float diff = dx - dx1;
	    if (fabs(diff) > thresh)
		continue; // fails cross checking -- occluded

	    mask.Pixel(x, y, 0) = 255; // cross-checking OK
	}
    }
}

int main(int argc, char *argv[])
{
    try {
	if (argc >= 5) {
	    // check if 3rd arg is .pfm, if so, interpret 2nd arg as ydisps
	    const char *dot = strrchr(argv[3], '.');
	    int ydisps = (dot != NULL && strcmp(dot, ".pfm") == 0);

	    int argn = 1;
	    char *disp0name = argv[argn++];
	    char *disp0yname = (ydisps? argv[argn++] : NULL);
	    char *disp1name = argv[argn++];
	    int dir = atoi(argv[argn++]);
	    char *maskname = argv[argn++];
	    float thresh = 1.0;
	    if (argn < argc)
		thresh = atof(argv[argn++]);

	    printf("thresh=%g\n", thresh);
      
	    CFloatImage fdisp0, fdisp1, fdisp0y;
	    CByteImage mask;
	    ReadImageVerb(fdisp0, disp0name, verbose);
	    if (ydisps) 
		ReadImageVerb(fdisp0y, disp0yname, verbose);
	    ReadImageVerb(fdisp1, disp1name, verbose);

	    makeNonoccMask(fdisp0, fdisp0y, fdisp1, dir, thresh, mask);

	    WriteImageVerb(mask, maskname, verbose);
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
