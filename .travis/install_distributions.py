import sys
try:
    import distributions
    assert distributions
    sys.exit(0)
except ImportError:
    pass

from subprocess import check_call
import os

DISTRIBUTIONS_GIT_REPO='https://github.com/stephentu/distributions.git'

check_call(['git', 'clone', DISTRIBUTIONS_GIT_REPO])
check_call(['make', 'protobuf'], cwd='distributions')

env = dict(os.environ)
env['DISTRIBUTIONS_USE_PROTOBUF'] = '1'
env['PYDISTRIBUTIONS_USE_LIB'] = '1'
check_call(['make', 'install_cy'], env=env, cwd='distributions')
