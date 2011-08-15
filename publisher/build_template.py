#!/usr/bin/env python

import os
import sys
import json
import tempita

if not len(sys.argv) > 1:
    print "Usage: build_template.py templatename configfile.json"
    sys.exit(-1)

template_fn = sys.argv[1]
config_fn = sys.argv[2]
if not (os.path.exists(template_fn) and os.path.exists(config_fn)):
    print "Cannot find files specified as parameters."
    sys.exit(-1)

template = tempita.Template(open(template_fn, 'r').read())
config = json.load(open(config_fn, 'r'))

print template.substitute(config)
