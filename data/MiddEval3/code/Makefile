SRC = ii.cpp evaldisp.cpp computemask.cpp pfm2png.cpp png2pgm.cpp disp2pfm.cpp

BIN = $(SRC:.cpp=)

IMGLIB = imageLib

DBG = -g
WARN = -W -Wall
OPT ?= -O3
CPPFLAGS = $(DBG) $(OPT) $(WARN) -I$(IMGLIB)
ARCH := $(shell arch)
LDLIBS = -L$(IMGLIB) -lImg.$(ARCH)$(DBG) -lpng -lz
EXE = $(SRC:.cpp=.exe)


all: $(BIN)


clean: 
	rm -f core* *.stackdump

allclean: clean
	rm -f $(BIN) $(EXE)
