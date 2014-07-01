#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from distutils.version import LooseVersion
from distutils.sysconfig import parse_makefile
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from cython import __version__ as cython_version

import numpy
import sys
import os

clang = False
if sys.platform.lower().startswith('darwin'):
    clang = True

# make sure C shared libraries exists
shared_libraries = [
    '../common/out/libmicroscopes_common.dylib' if clang else '../common/out/libmicroscopes_common.so',
    'out/libmicroscopes_mixturemodel.dylib' if clang else 'out/libmicroscopes_mixturemodel.so',
]

for so in shared_libraries:
    if not os.path.isfile(so):
        raise ValueError(
            "could not locate `{}'. make sure to run `make' first".format(so))

# append to library path
os.environ['LIBRARY_PATH'] = os.environ.get('LIBRARY_PATH', '') + ':../common/out:out'

min_cython_version = '0.20.2b1' if clang else '0.20.1'
if LooseVersion(cython_version) < LooseVersion(min_cython_version):
    raise ValueError(
        'cython support requires cython>={}'.format(min_cython_version))

distributions_inc, distributions_lib, debug_build = None, None, False
try:
    config = parse_makefile('../config.mk')
    distributions_inc = config.get('DISTRIBUTIONS_INC', None)
    distributions_lib = config.get('DISTRIBUTIONS_LIB', None)
    debug_build = config.get('DEBUG', 0) == 1
except IOError:
    pass

if distributions_inc is not None:
    print 'Using distributions_inc:', distributions_inc
if distributions_lib is not None:
    print 'Using distributions_lib:', distributions_lib
if debug_build:
    print 'Debug build'

extra_compile_args = ['-std=c++0x']
if clang:
    extra_compile_args.extend([
        '-mmacosx-version-min=10.7',  # for anaconda
        '-stdlib=libc++',
    ])
if debug_build:
    extra_compile_args.append('-DDEBUG_MODE')

extra_include_dirs = []
if distributions_inc is not None:
    extra_include_dirs.append(distributions_inc)

extra_link_args = []
if distributions_lib is not None:
    extra_link_args.extend([
        '-L' + distributions_lib,
        '-Wl,-rpath,' + distributions_lib
    ])

def make_extension(module_name):
    sources = [module_name.replace('.', '/') + '.pyx']
    return Extension(
        module_name,
        sources=sources,
        libraries=["microscopes_common", "microscopes_mixturemodel", "protobuf", "distributions_shared"],
        language="c++",
        include_dirs=[numpy.get_include(), '../common/include', 'include'] + extra_include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

extensions = cythonize([
    make_extension('microscopes.cxx.mixture.model'),
    make_extension('microscopes.cxx.mixture._model'),
], include_path=['../common'])

setup(ext_modules=extensions)
