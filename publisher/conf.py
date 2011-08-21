import glob
import os

output_dir    = '../output'
template_dir  = '_templates'
build_dir     = '_build'
bib_dir       = os.path.join(build_dir, 'bib')
toc_conf      = os.path.join(build_dir, 'toc.json')
proc_conf     = '../scipy_proc.json'
dirs          = sorted([d.split('/')[2] for d in glob.glob('%s/*' % output_dir) if os.path.isdir(d)])
