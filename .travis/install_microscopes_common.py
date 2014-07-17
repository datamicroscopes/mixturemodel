import sys
import os
try:
    import microscopes.cxx.common
    assert microscopes.cxx.common
    sys.exit(0)
except ImportError:
    pass

from subprocess import check_call
# assumes a git checkout of microscopes-common already exists,
# setup by before_install_microscopes_common.py
check_call(['make', 'travis_install'], cwd='common')
