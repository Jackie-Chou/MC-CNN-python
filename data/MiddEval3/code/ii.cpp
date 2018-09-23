// imginfo.cpp
//
// print information about input images to stdout

static const char *usage = "\n\
  usage: %s [-b band] [-m] img1 [img2 ...]\n\
\n\
  prints information about images given\n\
  -a     : use absolute values\n\
  -b B   : only print info about band B\n\
  -m     : only print min max for each band, followed by filename\n";

#include "imageLib.h"
#include <math.h>
#include <getopt.h>

#define VERBOSE 0

void infoByteImg(CByteImage im, int band, int minmax) {
    if (!minmax)
	printf("byte image\n");
    CShape sh = im.Shape();
    int width = sh.width, height = sh.height, nB = sh.nBands;
    if (!minmax)
	printf("\tmin\tmax\tavg\t%%0\t%%neg\t%%pos\tpix0,0\tcenter\n");
    for (int b=0; b<nB; b++) {
	if (band >= 0 && b != band)
	    continue;
	int min = 0;
	if (width > 0 && height > 0)
	    min = im.Pixel(0, 0, b);
	int max = min;
	double sum = 0;
	int zeros = 0;
	int nneg = 0;
	int npos = 0;
	for (int y=0; y<height; y++) {
	    for (int x=0; x<width; x++) {
		int v = im.Pixel(x, y, b);
		sum += v;
		min = __min(v, min);
		max = __max(v, max);
		if (v < 0) nneg++;
		if (v == 0) zeros++;
		if (v > 0) npos++;
	    }
	}
	int n = width * height;
	if (n != zeros + nneg + npos)
	    throw CError("can't be");
	float f = 100.0 / n;
	if (minmax)
	    printf("%3d %3d  ", min, max);
	else
	    printf("band %d:\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\n",
		   b, min, max,  sum/n, f*zeros, f*nneg, f*npos,
		   im.Pixel(0, 0, b),
		   im.Pixel(width/2, height/2, b));

    }
}

void infoFloatImg(CFloatImage im, int band, int minmax, int useabs) {
    if (!minmax)
	printf("float image\n");
    CShape sh = im.Shape();
    int width = sh.width, height = sh.height, nB = sh.nBands;
    if (!minmax)
	printf("\tmin\tmax\tavg\t%%0\t%%neg\t%%pos\t%%INF\t%%-INF\tNaN\tpix0,0\tcenter\n");
    for (int b=0; b<nB; b++) {
	if (band >= 0 && b != band)
	    continue;
	float min = INFINITY;
	float max = -INFINITY;
	double sum = 0;
	int zeros = 0;
	int nneg = 0;
	int npos = 0;
	int pinf = 0;
	int ninf = 0;
	int nnan = 0;
	int n = 0;
	for (int y=0; y<height; y++) {
	    for (int x=0; x<width; x++) {
		float v = im.Pixel(x, y, b);
		if (useabs)
		    v = fabs(v);
		if (isnan(v))
		    nnan++;
		else if (v == INFINITY)
		    pinf++;
		else if (v == -INFINITY)
		    ninf++;
		else {
		    sum += v;
		    n++;
		    min = __min(v, min);
		    max = __max(v, max);
		    if (v < 0) nneg++;
		    if (v == 0) zeros++;
		    if (v > 0) npos++;
		}
	    }
	}
	int nt = width * height;
	float f = 100.0 / nt;
	if (minmax)
	    printf("%12.5f %12.5f  ", min, max);
	else
	    printf("band %d:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f\t%.3f\n",
		   b, min, max,  sum/n, f*zeros, f*nneg, f*npos, f*pinf, f*ninf, nnan,
		   im.Pixel(0, 0, b),
		   im.Pixel(width/2, height/2, b));
    }
}

int main(int argc, char *argv[])
{
    int useabs = 0;
    int minmax = 0;
    int band = -1;
    
    int o;
    while ((o = getopt(argc, argv, "amb:")) != -1)
	switch (o) {
	case 'a': useabs = 1; break;
	case 'b': band = atoi(optarg); break;
	case 'm': minmax = 1; break;
	default: 
	    fprintf(stderr, "Ignoring unrecognized option\n");
	}
    if (optind == argc) {
	fprintf(stderr, usage, argv[0]);
	exit(1);
    }

    int nimgs = argc - optind;
    try {
	for (int i = 0; i < nimgs; i++) {
	    char *imgfile = argv[optind + i];
	    if (!minmax)
		printf("========= %s\n", imgfile);
	    CImage img;
	    ReadImageVerb(img, imgfile, VERBOSE);
	    CShape sh = img.Shape();
	    if (!minmax)
		printf("width = %d, height = %d, %.2f Mpixels, nBands = %d, ", 
		       sh.width, sh.height, sh.width*sh.height/1e6, sh.nBands);
	    if (img.PixType() == typeid(uchar)) {
		infoByteImg(*(CByteImage *) &img, band, minmax);
	    } else if (img.PixType() == typeid(float)) {
		infoFloatImg(*(CFloatImage *) &img, band, minmax, useabs);
	    } else
		throw CError("Illegal pixel type");
	    if (minmax)
		printf(" %s\n", imgfile);
	}
    }
    catch (CError &err) {
	fprintf(stderr, err.message);
	fprintf(stderr, "\n");
	return -1;
    }

    return 0;
}
