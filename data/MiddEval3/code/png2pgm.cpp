// png2pgm.cpp
//
// Reads image and writes as 1-band image (can use to change formats png -> pgm)

static const char *usage = "\n  usage: %s [-quiet] in out\n";

#include "imageLib.h"

int main(int argc, char *argv[])
{
    try {
	int quiet = 0;
	int argn = 1;
	if (argc > 1 && argv[1][0]=='-' && argv[1][1]=='q') {
	    quiet = 1;
	    argn++;
	}
	if ((argc==3 && !quiet) || (argc==4 && quiet)) {
	    CByteImage im;
	    ReadImageVerb (im, argv[argn++], !quiet);
	    im = ConvertToGray(im);
	    WriteImageVerb(im, argv[argn++], !quiet);
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
