import errno
import itertools
import os
import subprocess
import sys
import time
from argparse import Namespace

import pytest

from pyqtgraph.examples import utils


TIMEOUT_S = 10.0

def buildFileList(examples, files=None):
    examples_dir = os.path.abspath(os.path.dirname(utils.__file__))
    if files is None:
        files = []
    for key, val in examples.items():
        if isinstance(val, dict):
            buildFileList(val, files)
        elif isinstance(val, Namespace):
            files.append((key, os.path.join(examples_dir, val.filename)))
        else:
            files.append((key, os.path.join(examples_dir, val)))
    return files


path = os.path.abspath(os.path.dirname(__file__))
files = [("Example App", "RunExampleApp.py")]
for ex in [utils.examples_, utils.others]:
    files = buildFileList(ex, files)
files = sorted(set(files))

openglExamples = ['GLViewWidget.py']
openglExamples.extend(utils.examples_['3D Graphics'].values())
def testExamples(frontend, f):
    name, file = f
    global path
    fn = os.path.join(path, file)
    os.chdir(path)
    sys.stdout.write(f"{name}")
    sys.stdout.flush()
    import1 = "import %s" % frontend if frontend != '' else ''
    import2 = os.path.splitext(os.path.split(fn)[1])[0]
    timeout_ms = int(TIMEOUT_S * 1000)
    code = """
try:
    {0}
    import faulthandler
    faulthandler.enable()
    import pyqtgraph as pg
    import pyqtgraph.examples.{1} as {1}
    import sys
    print("test complete")
    sys.stdout.flush()
    pg.Qt.QtCore.QTimer.singleShot({2}, pg.Qt.QtWidgets.QApplication.quit)
    pg.exec()
    names = [x for x in dir({1}) if not x.startswith('_')]
    for name in names:
        delattr({1}, name)
except:
    print("test failed")
    raise

""".format(import1, import2, timeout_ms)
    env = dict(os.environ)
    example_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.dirname(os.path.dirname(example_dir))
    env['PYTHONPATH'] = f'{path}{os.pathsep}{example_dir}'
    process = subprocess.Popen([sys.executable],
                                stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                text=True,
                                env=env)
    process.stdin.write(code)
    process.stdin.close()

    output = ''
    fail = False
    while True:
        try:
            c = process.stdout.read(1)
        except IOError as err:
            if err.errno == errno.EINTR:
                # Interrupted system call; just try again.
                c = ''
            else:
                raise
        output += c
        if output.endswith('test complete'):
            break
        if output.endswith('test failed'):
            fail = True
            break
    start = time.time()
    killed = False
    while process.poll() is None:
        time.sleep(0.1)
        if time.time() - start > TIMEOUT_S and not killed:
            process.kill()
            killed = True

    stdout, stderr = (process.stdout.read(), process.stderr.read())
    process.stdout.close()
    process.stderr.close()

    if (fail or
        'Exception:' in stderr or
        'Error:' in stderr):
        if (not fail 
            and name == "RemoteGraphicsView" 
            and "pyqtgraph.multiprocess.remoteproxy.ClosedError" in stderr):
            # This test can intermittently fail when the subprocess is killed
            return None
        print(stdout)
        print(stderr)
        pytest.fail(
            f"{stdout}\n{stderr}\nFailed {name} Example Test Located in {file}",
            pytrace=False
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", "-t", type=float, default=TIMEOUT_S)
    parser.parse_args()
    TIMEOUT_S = parser.parse_args().timeout
    
    for file in itertools.cycle(files):
        try:
            testExamples("PySide6", file)
            print()
        except KeyboardInterrupt:
            break
        except:
            pass
