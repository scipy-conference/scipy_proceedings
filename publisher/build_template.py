#!/usr/bin/env python

import os
import sys
import shlex, subprocess

import tempita
from conf import bib_dir, build_dir, template_dir, html_dir
from options import get_config

class TeXTemplate(tempita.Template):
    def _repr(self, value, pos):
        if sys.version_info[0] < 3 and isinstance(value, unicode):
            value = value.replace('&', '\&')
        elif sys.version_info[0] >= 3 and isinstance(value, str):
            value = value.replace('&', '\&')
        else:
            value = str(value)
        return value.encode('utf-8')

def _from_template(tmpl_basename, config, use_html=True):
    tmpl = os.path.join(template_dir, tmpl_basename + '.tmpl')
    if use_html:
        template = tempita.HTMLTemplate(open(tmpl, 'r').read())
    else:
        template = TeXTemplate(open(tmpl, 'r').read())
    return template.substitute(config)

def from_template(tmpl_basename, config, dest_fn):
    extension = os.path.splitext(dest_fn)[1][1:]

    use_html = False if 'tex' in extension else True
    outfile = _from_template(tmpl_basename, config, use_html=use_html)
    outname = os.path.join(build_dir, extension, dest_fn)

    with open(outname, mode='w') as f:
        f.write(outfile)

def bib_from_tmpl(bib_type, config, target):
    tmpl_basename = bib_type + '.bib'
    dest_path = os.path.join(bib_dir, target + '.bib')
    from_template(tmpl_basename, config, dest_path)
    command_line = 'recode -d u8..ltex ' + dest_path
    run = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE)
    out, err = run.communicate()

def get_html_header(config):
    return _from_template('header.html', config)

def get_html_content(tmpl, config):
    return _from_template(tmpl, config)

def html_from_tmpl(src, config, target):

    header = get_html_header(config)
    content =  _from_template(src, config)

    outfile = header+content
    dest_fn = os.path.join(html_dir, target + '.html')
    extension = os.path.splitext(dest_fn)[1][1:]
    outname = os.path.join(build_dir, extension, dest_fn)
    with open(outname, mode='w') as f:
        f.write(outfile)

if __name__ == "__main__":

    if not len(sys.argv) == 2:
        print("Usage: build_template.py destination_name")
        sys.exit(-1)

    dest_fn = sys.argv[1]
    template_fn = os.path.join(template_dir, dest_fn+'.tmpl')

    if not os.path.exists(template_fn):
        print("Cannot find template.")
        sys.exit(-1)

    config = get_config()
    from_template(dest_fn, config, dest_fn)
