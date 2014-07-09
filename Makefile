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

TESTPROG_SRCFILES := $(wildcard test/cxx/*.cpp)
TESTPROG_BINFILES := $(patsubst %.cpp, %.prog, $(TESTPROG_SRCFILES))

TESTPROG_LDFLAGS := $(LDFLAGS)
TESTPROG_LDFLAGS += -L$(TOP)/out -Wl,-rpath,$(TOP)/out
TESTPROG_LDFLAGS += -lmicroscopes_mixturemodel

UNAME_S := $(shell uname -s)
TARGETS :=
LIBPATH_VARNAME :=
ifeq ($(UNAME_S),Linux)
	TARGETS := $(O)/libmicroscopes_mixturemodel.so
	LIBPATH_VARNAME := LD_LIBRARY_PATH
	EXTNAME := so
	SHARED_FLAG := -shared
endif
ifeq ($(UNAME_S),Darwin)
	TARGETS := $(O)/libmicroscopes_mixturemodel.dylib
	LIBPATH_VARNAME := DYLD_LIBRARY_PATH
	EXTNAME := dylib
	SHARED_FLAG := -dynamiclib
endif

all: $(TARGETS)

.PHONY: build_test_cxx
build_test_cxx: $(TESTPROG_BINFILES)

$(O)/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(O)/libmicroscopes_mixturemodel.$(EXTNAME): $(OBJFILES)
	$(CXX) $(SHARED_FLAG) -o $@ $(OBJFILES) $(LDFLAGS)

%.prog: %.cpp $(O)/libmicroscopes_mixturemodel.$(EXTNAME)
	$(CXX) $(CXXFLAGS) $< -o $@ $(TESTPROG_LDFLAGS)

DEPFILES := $(wildcard out/mixture/*.d)
ifneq ($(DEPFILES),)
-include $(DEPFILES)
endif

.PHONY: clean
clean: 
	rm -rf out test/cxx/*.{d,dSYM,prog}
	find microscopes \( -name '*.cpp' -or -name '*.so' -or -name '*.pyc' \) -type f -print0 | xargs -0 rm -f --

.PHONY: test
test: test_cxx
	python setup.py build_ext --inplace
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):../common/out:./out PYTHONPATH=$$PYTHONPATH:../common:. nosetests

.PHONY: test_cxx
test_cxx: build_test_cxx
	test/cxx/test_state.prog
