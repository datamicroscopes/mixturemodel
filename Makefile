all: release

.PHONY: release
release:
	@echo "Setting up cmake (release)"
	[ -d release ] || (mkdir release && cd release && eval `python ../cmake/print_cmake_command.py`)

.PHONY: debug
debug:
	@echo "Setting up cmake (debug)"
	[ -d debug ] || (mkdir debug && cd debug && eval `python ../cmake/print_cmake_command.py --debug`)

.PHONY: test
test:
	(cd test && nosetests --verbose)

.PHONY: travis_before_install
travis_before_install:
	(cd .travis && python before_install_microscopes_common.py)

COMPILER := CC=gcc-4.8 CXX=g++-4.8

.PHONY: travis_install
travis_install: 
	(cd .travis && python install_microscopes_common.py)
	pip install -r .travis/requirements.txt
	mkdir -p build
	(cd build && $(COMPILER) cmake -DCMAKE_INSTALL_PREFIX=$$VIRTUAL_ENV .. && make && make install)
	$(COMPILER) pip install .

.PHONY: travis_script
travis_script: 
	(cd build && make test)
	(cd test && nosetests --verbose -a '!slow')
