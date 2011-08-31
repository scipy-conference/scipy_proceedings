import glob
import os

work_dir      = os.path.dirname(__file__)
papers_dir    = os.path.join(work_dir,'../papers')
output_dir    = os.path.join(work_dir,'../output')
template_dir  = os.path.join(work_dir,'_templates')
build_dir     = os.path.join(work_dir,'_build')
bib_dir       = os.path.join(build_dir, 'bib')
pdf_dir       = os.path.join(build_dir, 'pdfs')
toc_conf      = os.path.join(build_dir, 'toc.json')
proc_conf     = os.path.join(work_dir,'../scipy_proc.json')
dirs          = sorted([os.path.basename(d)
                        for d in glob.glob('%s/*' % papers_dir)
                        if os.path.isdir(d)])
