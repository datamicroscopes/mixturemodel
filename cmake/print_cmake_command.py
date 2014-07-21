import sys
import os
from subprocess import check_output

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        debug = True
    else:
        debug = False
    ## XXX: handle virtualenv
    conda_full_path = check_output("which conda", shell=True).strip()
    debug_str = '-DCMAKE_BUILD_TYPE=Debug ' if debug else ''
    if 'CONDA_DEFAULT_ENV' in os.environ:
        a, b = os.path.split(conda_full_path)
        assert b == 'conda'
        a, b = os.path.split(a)
        assert b == 'bin'
        conda_env_path = a
        a, b = os.path.split(a)
        assert b == os.environ['CONDA_DEFAULT_ENV']
        print 'cmake {}-DCMAKE_INSTALL_PREFIX={} -DEXTRA_INCLUDE_PATH={} -DEXTRA_LIBRARY_PATH={} ..'.format(
                debug_str,
                conda_env_path,
                os.path.join(conda_env_path, 'include'),
                os.path.join(conda_env_path, 'lib'))
    else:
        a, b = os.path.split(conda_full_path)
        assert b == 'conda'
        a, b = os.path.split(a)
        assert b == 'bin'
        conda_env_path = a
        print 'cmake {}-DCMAKE_INSTALL_PREFIX={} -DEXTRA_INCLUDE_PATH={} -DEXTRA_LIBRARY_PATH={} ..'.format(
                debug_str,
                conda_env_path,
                os.path.join(conda_env_path, 'include'),
                os.path.join(conda_env_path, 'lib'))
