#!/usr/bin/env python


import os
import sys
import shlex
import subprocess
import io
import shutil


from distutils import dir_util

import tempita
from conf import (bib_dir, build_dir, template_dir, html_dir,
                  static_dir, status_file)
from options import get_config

class TeXTemplate(tempita.Template):
    def _repr(self, value, pos):
        if sys.version_info[0] < 3 and isinstance(value, unicode):
            value = value.replace('&', '\&')
        elif sys.version_info[0] >= 3 and isinstance(value, str):
            value = value.replace('&', '\&')
        elif sys.version_info[0] < 3 :
            value = unicode(value)
        else:
            value = str(value)
        return value

def _from_template(tmpl_basename, config, template_type='raw'):
    tmpl = os.path.join(template_dir, tmpl_basename + '.tmpl')
    if template_type=='html':
        with io.open(tmpl, mode='r', encoding='utf-8') as f:
            template = tempita.HTMLTemplate(f.read())
    elif template_type=='latex':
        with io.open(tmpl, mode='r', encoding='utf-8') as f:
            template = TeXTemplate(f.read())
    else:
        with io.open(tmpl, mode='r', encoding='utf-8') as f:
            template = tempita.Template(f.read())
    return template.substitute(config)

def from_template(tmpl_basename, config, dest_fn):
    extension = os.path.splitext(dest_fn)[1][1:]

    template_type = 'tex' if 'tex' in extension else 'html'
    outfile = _from_template(tmpl_basename, config, template_type=template_type)
    outname = os.path.join(build_dir, extension, dest_fn)

    with io.open(outname, mode='w', encoding='utf-8') as f:
        f.write(outfile)

def bib_from_tmpl(bib_type, config, target):
    tmpl_basename = bib_type + '.bib'
    dest_path = os.path.join(bib_dir, target + '.bib')
    from_template(tmpl_basename, config, dest_path)
    command_line = 'recode -d u8..ltex ' + dest_path
    run = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE)
    out, err = run.communicate()

def get_html_header(config):
    return _from_template('header.html', config, template_type='html')

def get_html_content(tmpl, config):
    return _from_template(tmpl, config, template_type='html')

def html_from_tmpl(src, config, target):

    header = get_html_header(config)
    content =  _from_template(src, config, template_type='html')

    outfile = header+content
    dest_fn = os.path.join(html_dir, target + '.html')
    extension = os.path.splitext(dest_fn)[1][1:]
    outname = os.path.join(build_dir, extension, dest_fn)
    with io.open(outname, mode='w', encoding='utf-8') as f:
        f.write(outfile)

def copy_static_files(dest_fn):
    extension = os.path.splitext(dest_fn)[1][1:]
    outdir = os.path.join(build_dir, extension, "static")
    dir_util.copy_tree(static_dir, outdir)
    style_fn = os.path.join(outdir, 'status.sty')
    shutil.copy(status_file, style_fn)

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
    copy_static_files(dest_fn)
