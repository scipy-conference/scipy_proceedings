import docutils.core as dc
from writer import writer

settings = {'documentclass': 'IEEEtran'}

f = open('papers/00_vanderwalt/00_vanderwalt.rst', 'r')
tex = dc.publish_string(source=f.read(), writer=writer,
                        settings_overrides=settings)

out = open('/tmp/paper.tex', 'w')
out.write(tex)
out.close()

