import os
import shutil
import tempfile
import subprocess
from subprocess import PIPE

MODE_BATCH = 0
MODE_NON_STOP = 1
MODE_SCROLL = 2
MODE_ERROR_STOP = 3
INTERACTION_MODES = ['batchmode', 'nonstopmode', 'scrollmode', 'errorstopmode']

JINJA2_ENV = {'block_start_string': '\BLOCK{',
              'block_end_string': '}',
              'variable_start_string': '\VAR{',
              'variable_end_string': '}',
              'comment_start_string': '\#{',
              'comment_end_string': '}',
              'line_statement_prefix': '%%',
              'line_comment_prefix': '%#',
              'trim_blocks': True,
              'autoescape': False}


class PDFLaTeX:
    def __init__(self, latex_src, job_name: str):
        self.latex = latex_src
        self.job_name = job_name
        self.interaction_mode = INTERACTION_MODES[MODE_BATCH]
        self.dir = None
        self.pdf_filename = None
        self.params = dict()
        self.pdf = None
        self.log = None
        
    @classmethod
    def from_texfile(cls, filename):
        prefix = os.path.basename(filename)
        prefix = os.path.splitext(prefix)[0]
        with open(filename, 'rb') as f:
            return cls.from_binarystring(f.read(), prefix)

    @classmethod
    def from_binarystring(cls, binstr: str, jobname: str):
        return cls(binstr, jobname)

    @classmethod
    def from_jinja_template(cls, jinja2_template, jobname: str = None, **render_kwargs):
        tex_src = jinja2_template.render(**render_kwargs)
        fn = jinja2_template.filename
        
        if fn is None:
            fn = jobname
            if not fn:
                raise ValueError("PDFLaTeX: if jinja template does not have attribute 'filename' set, "
                                 "'jobname' must be provided")
        return cls(tex_src, fn)

    def create_pdf(self, keep_pdf_file: bool = False, keep_log_file: bool = False, env: dict = None):
        if self.interaction_mode is not None:
            self.add_args({'-interaction-mode': self.interaction_mode})
        
        dir = self.params.get('-output-directory')
        filename = self.params.get('-jobname')
        
        if filename is None:
            filename = self.job_name
        if dir is None:
            dir = ""
        
        with tempfile.TemporaryDirectory() as td:
            self.set_output_directory(td)
            self.set_jobname('file')
    
            args = self.get_run_args()
            fp = subprocess.run(args, input=self.latex, env=env, timeout=15, stdout=PIPE, stderr=PIPE)
            with open(os.path.join(td, 'file.pdf'), 'rb') as f:
                self.pdf = f.read()
            with open(os.path.join(td, 'file.log'), 'rb') as f:
                self.log = f.read()
            if keep_log_file:
                shutil.move(os.path.join(td, 'file.log'), os.path.join(dir, filename + '.log'))
            if keep_pdf_file:
                shutil.move(os.path.join(td, 'file.pdf'), os.path.join(dir, filename + '.pdf'))
        
        return self.pdf, self.log, fp

    def get_run_args(self):
        a = [k+('='+v if v is not None else '') for k, v in self.params.items()]
        a.insert(0, 'pdflatex')
        return a
    
    def add_args(self, params: dict):
        for k in params:
            self.params[k] = params[k]
    
    def del_args(self, params):
        if isinstance(params, str):
            params = [params]

        if isinstance(params, dict) or isinstance(params, list):
            for k in params:
                if k in self.params.keys():
                    del self.params[k]
        else:
            raise ValueError('PDFLaTeX: del_cmd_params: parameter must be str, dict or list.')
    
    def set_output_directory(self, dir: str = None):
        self.generic_param_set('-output-directory', dir)

    def set_jobname(self, jobname: str = None):
        self.generic_param_set('-jobname', jobname)

    def set_output_format(self, fmt: str = None):
        if fmt and fmt not in ['pdf', 'dvi']:
            raise ValueError("PDFLaTeX: Format must be either 'pdf' or 'dvi'.")
        self.generic_param_set('-output-format', dir)
    
    def generic_param_set(self, param_name, value):
        if value is None:
            if param_name in self.params.keys():
                del self.params[param_name]
        else:
            self.params[param_name] = value
    
    def set_pdf_filename(self, pdf_filename: str = None):
        self.set_jobname(pdf_filename)
    
    def set_batchmode(self, on: bool = True):
        self.interaction_mode = INTERACTION_MODES[MODE_BATCH] if on else None

    def set_nonstopmode(self, on: bool =True):
        self.interaction_mode = INTERACTION_MODES[MODE_NON_STOP] if on else None

    def set_scrollmode(self, on: bool = True):
        self.interaction_mode = INTERACTION_MODES[MODE_SCROLL] if on else None

    def set_errorstopmode(self, on: bool = True):
        self.interaction_mode = INTERACTION_MODES[MODE_ERROR_STOP] if on else None

    def set_interaction_mode(self, mode: int = None):
        if mode is None:
            self.interaction_mode = None
        elif 0 <= mode <= 3:
            self.interaction_mode = INTERACTION_MODES[mode]
        else:
            raise ValueError('PDFLaTeX: Invalid interaction mode!')