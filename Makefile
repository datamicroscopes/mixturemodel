-include config.mk

O := out
TOP := $(shell echo $${PWD-`pwd`})

DEBUG ?= 0
MICROSCOPES_COMMON_REPO ?= $(TOP)/../common

# set the CXXFLAGS
CXXFLAGS := -fPIC -g -MD -Wall -std=c++0x -I$(TOP)/include
CXXFLAGS += -I$(MICROSCOPES_COMMON_REPO)/include
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
LDFLAGS += -L$(MICROSCOPES_COMMON_REPO)/out -Wl,-rpath,$(MICROSCOPES_COMMON_REPO)/out
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
	SOFLAGS := -shared
endif
ifeq ($(UNAME_S),Darwin)
	TARGETS := $(O)/libmicroscopes_mixturemodel.dylib
	LIBPATH_VARNAME := DYLD_LIBRARY_PATH
	EXTNAME := dylib
	SOFLAGS := -dynamiclib -install_name $(TOP)/$(O)/libmicroscopes_mixturemodel.$(EXTNAME)
endif

all: $(TARGETS)

.PHONY: build_test_cxx
build_test_cxx: $(TESTPROG_BINFILES)

$(O)/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(O)/libmicroscopes_mixturemodel.$(EXTNAME): $(OBJFILES)
	$(CXX) -o $@ $(OBJFILES) $(LDFLAGS) $(SOFLAGS)

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
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):$(MICROSCOPES_COMMON_REPO)/out:./out PYTHONPATH=$$PYTHONPATH:$(MICROSCOPES_COMMON_REPO):. nosetests

.PHONY: test_cxx
test_cxx: build_test_cxx
	test/cxx/test_state.prog
