// pfm2png.cpp 

// converts pfm disparities to color png for visualization
// maps INFINITY (unknown) to 0

// DS 6/9/2014

static const char *usage = "\n\
  usage: %s [-c calib.txt] [-m dmin] [-d dmax] disp.pfm disp.png\n\
\n\
  maps float disparities from dmin..dmax to colors using 'jet' color map\n\
  unknown values INF are mapped to black\n\
  dmin, dmax can be supplied or read from calib file,\n\
  otherwise, min and max values of input data are used\n";

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include "imageLib.h"

// read vmin, vmax from calib file
void readVizRange(char *calibfile, float& dmin, float& dmax)
{
    char line[1000];
    float f;

    dmin = INFINITY;
    dmax = INFINITY;
    FILE *fp = fopen(calibfile, "r");
    if (fp != NULL ) {
	while (fgets(line, sizeof line, fp) != NULL ) {
	    if (sscanf(line, " vmin= %f", &f) == 1) dmin = f;
	    if (sscanf(line, " vmax= %f", &f) == 1) dmax = f;
	}
	fclose (fp);
    } else 
	throw CError("Cannot open calib file %s\n", calibfile);
    if (dmin == INFINITY || dmax == INFINITY)
	throw CError("Cannot extract vmin, vmax from calib file %s\n", calibfile);
    //printf("read vmin=%f, vmax=%f\n", dmin, dmax);
}

// translate value x in [0..1] into color triplet using "jet" color map
// if out of range, use darker colors
// variation of an idea by http://www.metastine.com/?p=7
void jet(float x, int& r, int& g, int& b)
{
    if (x < 0) x = -0.05;
    if (x > 1) x =  1.05;
    x = x / 1.15 + 0.1; // use slightly asymmetric range to avoid darkest shades of blue.
    r = __max(0, __min(255, (int)(round(255 * (1.5 - 4*fabs(x - .75))))));
    g = __max(0, __min(255, (int)(round(255 * (1.5 - 4*fabs(x - .5))))));
    b = __max(0, __min(255, (int)(round(255 * (1.5 - 4*fabs(x - .25))))));
}


// get min and max (non-INF) values
void getMinMax(CFloatImage fimg, float& vmin, float& vmax)
{
    CShape sh = fimg.Shape();
    int width = sh.width, height = sh.height;

    vmin = INFINITY;
    vmax = -INFINITY;

    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	    float f = fimg.Pixel(x, y, 0);
	    if (f == INFINITY)
		continue;
	    vmin = min(f, vmin);
	    vmax = max(f, vmax);
	}
    }
}

// convert float disparity image into a color image using jet colormap
void float2color(CFloatImage fimg, CByteImage &img, float dmin, float dmax)
{
    CShape sh = fimg.Shape();
    int width = sh.width, height = sh.height;
    sh.nBands = 3;
    img.ReAllocate(sh);

    float scale = 1.0 / (dmax - dmin);

    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	    float f = fimg.Pixel(x, y, 0);
	    int r = 0;
	    int g = 0;
	    int b = 0;
	    
	    if (f != INFINITY) {
		float val = scale * (f - dmin);
		jet(val, r, g, b);
	    }

	    img.Pixel(x, y, 0) = b;
	    img.Pixel(x, y, 1) = g;
	    img.Pixel(x, y, 2) = r;
	}
    }
}

int main(int argc, char *argv[])
{
    char *calibfile = NULL;
    float dmin = 0;
    float dmax = 0;
    int verbose = 0;

    int o;
    while ((o = getopt(argc, argv, "c:m:d:")) != -1)
	switch (o) {
	case 'c': calibfile  = optarg; break;
	case 'm': dmin  = atof(optarg); break;
	case 'd': dmax  = atof(optarg); break;
	default: 
	    fprintf(stderr, "Ignoring unrecognized option\n");
	}
    if (optind != argc-2) {
	fprintf(stderr, usage, argv[0]);
	exit(1);
    }
    try {
	if (calibfile != NULL)
	    readVizRange(calibfile, dmin, dmax); // if calibfile given, extract dmin, dmax from that
	    	
	char *fdispname = argv[optind++];
	char *dispname = argv[optind++];
      
	CFloatImage fdisp;
	CByteImage disp;
	ReadImageVerb(fdisp, fdispname, verbose);
	if (dmax == dmin)
	    getMinMax(fdisp, dmin, dmax);

	//printf("dmin=%f, dmax=%f\n", dmin, dmax);

	float2color(fdisp, disp, dmin, dmax);
	WriteImageVerb(disp, dispname, verbose);
    }
    catch (CError &err) {
	fprintf(stderr, err.message);
	fprintf(stderr, "\n");
	return -1;
    }
  
    return 0;
}
