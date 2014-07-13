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

min_cython_version = '0.20.2b1' if clang else '0.20.1'
if LooseVersion(cython_version) < LooseVersion(min_cython_version):
    raise ValueError(
        'cython support requires cython>={}'.format(min_cython_version))

so_ext = 'dylib' if clang else 'so'

KEYS = (
    'DISTRIBUTIONS_INC',
    'DISTRIBUTIONS_LIB',
    'CC',
    'CXX',
    'DEBUG',
    'MICROSCOPES_COMMON_REPO',
    )

def get_config_info(config):
    config = parse_makefile(config)
    ret = {}
    for k in KEYS:
        if k in config:
            ret[k] = config[k]
    return ret

def merge_config(existing, overwriting):
    existing.update(overwriting)

config = {}
for fname in ('../config.mk', 'config.mk'):
    try:
        merge_config(config, get_config_info(fname))
    except IOError:
        pass

distributions_inc = config.get('DISTRIBUTIONS_INC', None)
distributions_lib = config.get('DISTRIBUTIONS_LIB', None)
cc = config.get('CC', None)
cxx = config.get('CXX', None)
debug_build = config.get('DEBUG', 0) == 1
microscopes_common_repo = config.get('MICROSCOPES_COMMON_REPO', None)

if distributions_inc is not None:
    print 'Using distributions_inc:', distributions_inc
if distributions_lib is not None:
    print 'Using distributions_lib:', distributions_lib
if cc is not None:
    print 'Using CC={}'.format(cc)
    os.environ['CC'] = cc
if cxx is not None:
    print 'Using CXX={}'.format(cxx)
    os.environ['CXX'] = cxx
if debug_build:
    print 'Debug build'
if microscopes_common_repo:
    print 'Using microscopes_common_repo:', microscopes_common_repo
else:
    microscopes_common_repo = '../common' # default path

# check that microscopes_common_repo is a valid dir
if not os.path.isdir(microscopes_common_repo):
    raise ValueError('not a valid directory: {}'.format(microscopes_common_repo))

# make sure C shared libraries exists
shared_libraries = (
    '{}/out/libmicroscopes_common.{}'.format(microscopes_common_repo, so_ext),
    'out/libmicroscopes_mixturemodel.{}'.format(so_ext),
)

for so in shared_libraries:
    if not os.path.isfile(so):
        raise ValueError(
            "could not locate `{}'. make sure to run `make' first".format(so))

# append to library path
os.environ['LIBRARY_PATH'] = \
        os.environ.get('LIBRARY_PATH', '') + \
        ':{}/out:out'.format(microscopes_common_repo)

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
        include_dirs=[numpy.get_include(), '{}/include'.format(microscopes_common_repo), 'include'] + extra_include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

extensions = cythonize([
    make_extension('microscopes.cxx.mixture.model'),
    make_extension('microscopes.cxx.mixture._model'),
], include_path=[microscopes_common_repo])

setup(ext_modules=extensions)
