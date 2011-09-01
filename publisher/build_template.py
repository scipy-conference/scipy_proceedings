#!/usr/bin/env python

import os
import sys
import shlex, subprocess

import tempita
from conf import bib_dir, build_dir, template_dir, html_dir
from options import get_config

def from_template(template_fn, config, dest_fn):

    template = tempita.HTMLTemplate(open(template_fn, 'r').read())
    
    extension = os.path.splitext(dest_fn)[1][1:]
    outname = os.path.join(build_dir, extension, dest_fn) 
    mode = 'w'
    
    with open(outname, mode=mode) as f:
        f.write(template.substitute(config))

def bib_from_tmpl(bib_type, config, target):
    bib_tmpl = os.path.join(template_dir, bib_type + '.bib.tmpl')
    dest_path = os.path.join(bib_dir, target + '.bib')
    from_template(bib_tmpl, config, dest_path)
    command_line = 'recode -d u8..ltex ' + dest_path
    run = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE)
    out, err = run.communicate()

def art_from_tmpl(config, target):
    art_tmpl = os.path.join(template_dir, 'article.html.tmpl')
    dest_path = os.path.join(html_dir, target + '.html')
    from_template(art_tmpl, config, dest_path)

if __name__ == "__main__":

    if not len(sys.argv) == 2:
        print "Usage: build_template.py destination_name"
        sys.exit(-1)
    
    dest_fn = sys.argv[1]
    template_fn = os.path.join(template_dir, dest_fn+'.tmpl')
    
    if not os.path.exists(template_fn):
        print "Cannot find template."
        sys.exit(-1)

    config = get_config()
    from_template(template_fn, config, dest_fn)
