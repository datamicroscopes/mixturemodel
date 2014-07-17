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
