import docutils.core as dc
from writer import writer

f = open('papers/00_vanderwalt/00_vanderwalt.rst', 'r')
print dc.publish_string(source=f.read(), writer=writer)
