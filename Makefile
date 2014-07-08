-include config.mk

DEBUG ?= 0

O := out
TOP := $(shell echo $${PWD-`pwd`})

# set the CXXFLAGS
CXXFLAGS := -fPIC -g -MD -Wall -std=c++0x -I$(TOP)/include
CXXFLAGS += -I$(TOP)/../common/include
ifneq ($(strip $(DEBUG)),1)
	CXXFLAGS += -O3 -DNDEBUG
else
	CXXFLAGS += -DDEBUG_MODE
endif
ifneq ($(strip $(DISTRIBUTIONS_INC)),)
	CXXFLAGS += -I$(DISTRIBUTIONS_INC)
endif

# set the LDFLAGS
LDFLAGS := -lprotobuf -ldistributions_shared -lmicroscopes_common
LDFLAGS += -L$(TOP)/../common/out -Wl,-rpath,$(TOP)/../common/out
ifneq ($(strip $(DISTRIBUTIONS_LIB)),)
	LDFLAGS += -L$(DISTRIBUTIONS_LIB) -Wl,-rpath,$(DISTRIBUTIONS_LIB) 
endif

SRCFILES := $(wildcard src/mixture/*.cpp) 
OBJFILES := $(patsubst src/%.cpp, $(O)/%.o, $(SRCFILES))

UNAME_S := $(shell uname -s)
TARGETS :=
LIBPATH_VARNAME :=
ifeq ($(UNAME_S),Linux)
	TARGETS := $(O)/libmicroscopes_mixturemodel.so
	LIBPATH_VARNAME := LD_LIBRARY_PATH
	EXTNAME := so
endif
ifeq ($(UNAME_S),Darwin)
	TARGETS := $(O)/libmicroscopes_mixturemodel.dylib
	LIBPATH_VARNAME := DYLD_LIBRARY_PATH
	EXTNAME := dylib
endif

all: $(TARGETS)

$(O)/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(O)/libmicroscopes_mixturemodel.so: $(OBJFILES)
	gcc -shared -o $(O)/libmicroscopes_mixturemodel.so $(OBJFILES) $(LDFLAGS)

$(O)/libmicroscopes_mixturemodel.dylib: $(OBJFILES)
	g++ -dynamiclib -o $(O)/libmicroscopes_mixturemodel.dylib $(OBJFILES) $(LDFLAGS)

DEPFILES := $(wildcard out/mixture/*.d)
ifneq ($(DEPFILES),)
-include $(DEPFILES)
endif

.PHONY: clean
clean: 
	rm -rf out test/cxx/*.{d,prog}
	find microscopes \( -name '*.cpp' -or -name '*.so' -or -name '*.pyc' \) -type f -print0 | xargs -0 rm --

.PHONY: test
test:
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):../common/out:./out PYTHONPATH=$$PYTHONPATH:../common:. nosetests

# test progs
PROGS := $(wildcard test/cxx/*.cpp)
OUTFILES := $(patsubst %.cpp, %.prog, $(PROGS))

PROG_LDFLAGS := $(LDFLAGS)
PROG_LDFLAGS += -L$(TOP)/out -Wl,-rpath,$(TOP)/out
PROG_LDFLAGS += -lmicroscopes_mixturemodel

%.prog: %.cpp $(O)/libmicroscopes_mixturemodel.$(EXTNAME)
	$(CXX) $(CXXFLAGS) $< -o $@ $(PROG_LDFLAGS)

.PHONY: test_cxx
test_cxx: $(OUTFILES)
