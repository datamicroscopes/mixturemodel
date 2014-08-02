#!/bin/sh
set -e

UNAME=`uname`
if [ "${UNAME}" = "Darwin" ]; then
    export EXTRA_LINK_ARGS=-headerpad_max_install_names
elif [ "${UNAME}" = "Linux" ]; then
    if (which g++-4.8 >/dev/null 2>&1); then
      export CXX=g++-4.8
    fi
    if (which gcc-4.8 >/dev/null 2>&1); then
      export CC=gcc-4.8
    fi
else
    echo "unsupported os: ${UNAME}"
    exit 1
fi

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} -DEXTRA_INCLUDE_PATH=${PREFIX}/include -DEXTRA_LIBRARY_PATH=${PREFIX}/lib ..
make VERBOSE=1 && make install
cd ..
OFFICIAL_BUILD=1 LIBRARY_PATH=${PREFIX}/lib EXTRA_INCLUDE_PATH=${PREFIX}/include $PYTHON setup.py install
