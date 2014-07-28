all: release

.PHONY: release
release:
	@echo "Setting up cmake (release)"
	@python ./cmake/print_cmake_command.py Release
	[ -d release ] || (mkdir release && cd release && eval `python ../cmake/print_cmake_command.py Release`)

.PHONY: relwithdebinfo
relwithdebinfo:
	@echo "Setting up cmake (relwithdebinfo)"
	@python ./cmake/print_cmake_command.py RelWithDebInfo
	[ -d relwithdebinfo ] || (mkdir relwithdebinfo && cd relwithdebinfo && eval `python ../cmake/print_cmake_command.py RelWithDebInfo`)

.PHONY: debug
debug:
	@echo "Setting up cmake (debug)"
	@python ./cmake/print_cmake_command.py Debug
	[ -d debug ] || (mkdir debug && cd debug && eval `python ../cmake/print_cmake_command.py Debug`)

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
	(cd build && CTEST_OUTPUT_ON_FAILURE=true make test)
	(cd test && nosetests --verbose -a '!slow')
