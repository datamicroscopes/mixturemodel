import sys
import os
try:
    import microscopes.cxx.common
    assert microscopes.cxx.common
    sys.exit(0)
except ImportError:
    pass

READ_ONLY_USERNAME = 'datamicroscopes-travis-builder'
READ_ONLY_PASSWORD = '458fa9be7190e08dd0aa328fbd95d8756e15cded8d2a56e1634f606702337b3db1cd6eb6dee4cdfa0da41a270aa92befba19'

MICROSCOPES_COMMON_GIT_REPO = \
    'https://{}:{}@github.com/datamicroscopes/common.git'.format(
	READ_ONLY_USERNAME, READ_ONLY_PASSWORD)

from subprocess import check_call
check_call(['git', 'clone', MICROSCOPES_COMMON_GIT_REPO])
check_call(['make', 'travis_before_install'], cwd='common')
