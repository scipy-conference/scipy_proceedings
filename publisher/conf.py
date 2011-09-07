import glob
import os

work_dir      = os.path.dirname(__file__)
papers_dir    = os.path.join(work_dir,'../papers')
output_dir    = os.path.join(work_dir,'../output')
template_dir  = os.path.join(work_dir,'_templates')
static_dir    = os.path.join(work_dir,'_static')
css_file      = os.path.join(static_dir,'scipy-proc.css')
toc_list      = os.path.join(static_dir,'toc.txt')
build_dir     = os.path.join(work_dir,'_build')
pdf_dir       = os.path.join(build_dir, 'pdfs')
html_dir      = os.path.join(build_dir, 'html')
bib_dir       = os.path.join(html_dir, 'bib')
toc_conf      = os.path.join(build_dir, 'toc.json')
proc_conf     = os.path.join(work_dir,'../scipy_proc.json')
if os.path.isfile(toc_list):
    with open(toc_list) as f:
        dirs  = f.read().splitlines()
else:
    dirs      = sorted([os.path.basename(d)
                        for d in glob.glob('%s/*' % papers_dir)
                        if os.path.isdir(d)])
