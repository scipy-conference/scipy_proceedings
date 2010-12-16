import docutils.core as dc

from writer import writer

import os.path
import sys
import glob

preamble = '\n'.join([r'% PDF Standard Fonts',
                      r'\usepackage{mathptmx}',
                      r'\usepackage[scaled=.80]{helvet}',
                      r'\usepackage{courier}'],)

settings = {'documentclass': 'IEEEtran',
            'use_verbatim_when_possible': True,
            'use_latex_citations': True,
            'latex_preamble': preamble}

if len(sys.argv) != 2:
    print "Usage: build_paper.py paper_directory"
    sys.exit(-1)

path = sys.argv[1]
if not os.path.isdir(path):
    print("Cannot open directory: %s" % path)
    sys.exit(-1)

rst = glob.glob(os.path.join(path, '*.rst'))[0]

content = open(rst, 'r').read()
content = '''
.. role:: math(raw)
   :format: latex

''' + content

tex = dc.publish_string(source=content, writer=writer,
                        settings_overrides=settings)

out = open('/tmp/paper.tex', 'w')
out.write(tex)
out.close()

