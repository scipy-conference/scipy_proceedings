import docutils.core as dc
from writer import writer

import os.path
import sys
import glob

settings = {'documentclass': 'IEEEtran'}

if len(sys.argv) != 2:
    print "Usage: build_paper.py paper_directory"
    sys.exit(-1)

path = sys.argv[1]
if not os.path.isdir(path):
    print("Cannot open directory: %s" % path)
    sys.exit(-1)

rst = glob.glob(os.path.join(path, '*.rst'))[0]

f = open(rst, 'r')
tex = dc.publish_string(source=f.read(), writer=writer,
                        settings_overrides=settings)

out = open('/tmp/paper.tex', 'w')
out.write(tex)
out.close()

