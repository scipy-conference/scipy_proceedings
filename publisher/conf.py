import glob
import os

output_dir = 'output'
cover_dir = 'cover_material'
dirs = sorted([d.split('/')[1] for d in glob.glob('%s/*' % output_dir) if os.path.isdir(d)])
