import glob
import os

output_dir    = '../output'
template_dir  = '_templates'
build_dir     = '_build'
dirs          = sorted([d.split('/')[2] for d in glob.glob('%s/*' % output_dir) if os.path.isdir(d)])
