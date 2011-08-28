#!/usr/bin/env python

import os
import sys

import tempita
import conf
from options import get_config

template_dir = conf.template_dir
build_dir    = conf.build_dir

if not len(sys.argv) == 2:
    print "Usage: build_template.py destination_name"
    sys.exit(-1)

dest_fn = sys.argv[1]
template_fn = os.path.join(template_dir, dest_fn+'.tmpl')

if not os.path.exists(template_fn):
    print "Cannot find template."
    sys.exit(-1)

template = tempita.HTMLTemplate(open(template_fn, 'r').read())
config = get_config()

extension = os.path.splitext(dest_fn)[1][1:]
outname = os.path.join(build_dir, extension, dest_fn) 
mode = 'w'

with open(outname, mode=mode) as f:
    f.write(template.substitute(config))
