import glob
import os
import io

# to test the build process on the example papers / slides
# comment out the entries in the excludes list
excludes = [
    'example',
    '00_myst_template',
    '00_tex_template',
]
# status_file_root possible values: draft, conference, ready
status_file_base = 'draft'
status_file_name = ''.join([status_file_base, '.sty'])

work_dir      = os.path.dirname(__file__)
papers_dir    = os.path.join(work_dir, '../papers')
present_dir   = os.path.join(work_dir, '../presentations')
keynote_dir   = os.path.join(present_dir, 'keynote')
slides_dir    = os.path.join(present_dir, 'slides')
posters_dir   = os.path.join(present_dir, 'posters')
lightning_dir = os.path.join(present_dir, 'lightning')
tools_dir     = os.path.join(present_dir, 'tools')
output_dir    = os.path.join(work_dir, '../output')
template_dir  = os.path.join(work_dir, '_templates')
static_dir    = os.path.join(work_dir, '_static')
css_file      = os.path.join(static_dir, 'scipy-proc.css')
toc_list      = os.path.join(static_dir, 'toc.txt')
build_dir     = os.path.join(work_dir, '_build')
pdf_dir       = os.path.join(build_dir, 'pdfs')
html_dir      = os.path.join(build_dir, 'html')
bib_dir       = os.path.join(html_dir, 'bib')
toc_conf      = os.path.join(build_dir, 'toc.json')
proc_conf     = os.path.join(work_dir, '../scipy_proc.json')
xref_conf     = os.path.join(build_dir, 'doi_batch')
other_conf    = os.path.join(build_dir, 'other.json')
status_file   = os.path.join(static_dir, status_file_name)

if os.path.isfile(toc_list):
    with io.open(toc_list, 'r', encoding='utf-8') as f:
        dirs = f.read().splitlines()
else:
    dirs = sorted([os.path.basename(d)
                   for d in glob.glob('%s/*' % papers_dir)
                   if os.path.isdir(d) and not any(e in d for e in excludes)])

other_dirs = {dir: sorted([os.path.basename(subdir)
                     for subdir in glob.glob('%s/*' % dir)
                     if os.path.isdir(subdir) and not any(e in subdir for e in excludes)])
                     for dir in (keynote_dir, slides_dir, posters_dir, lightning_dir, tools_dir)}

