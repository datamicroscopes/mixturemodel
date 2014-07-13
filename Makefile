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

.PHONY: build_py
build_py: $(O)/libmicroscopes_mixturemodel.$(EXTNAME)
	python setup.py build_ext --inplace

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
test: build_py test_cxx
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):$(MICROSCOPES_COMMON_REPO)/out:./out PYTHONPATH=$$PYTHONPATH:$(MICROSCOPES_COMMON_REPO):. nosetests

.PHONY: test_cxx
test_cxx: build_test_cxx
	test/cxx/test_state.prog

READ_ONLY_USERNAME=datamicroscopes-travis-builder
READ_ONLY_PASSWORD=458fa9be7190e08dd0aa328fbd95d8756e15cded8d2a56e1634f606702337b3db1cd6eb6dee4cdfa0da41a270aa92befba19

.PHONY: travis_before_install
travis_before_install:
	git clone https://$(READ_ONLY_USERNAME):$(READ_ONLY_PASSWORD)@github.com/datamicroscopes/common.git .travis/common
	$(MAKE) -C .travis/common travis_before_install
	sudo apt-get install -qq libblas-dev liblapack-dev

.PHONY: travis_install
travis_install: 
	$(MAKE) -C .travis/common travis_install
	pip install -r .travis/requirements.txt
	cp .travis/config.mk .
	echo "DISTRIBUTIONS_LIB = $$VIRTUAL_ENV/lib" >> config.mk
	make build_py build_test_cxx

.PHONY: travis_script
travis_script: test
