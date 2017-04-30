#!/usr/bin/env python
from __future__ import print_function

import glob
import shutil
import io
import os
import nbformat

from traitlets.config import Config
from nbconvert import RSTExporter
from nbconvert.writers import FilesWriter

from conf import papers_dir, output_dir
from utils import glob_for_one_file

class NotebookConverter(object):
    
    def __init__(self, config=None, paper_id='', keep_rst=False, debug=False):
        
        self.paper_id = paper_id
        self.in_path = os.path.join(papers_dir, self.paper_id)
        self.out_path = os.path.join(output_dir, self.paper_id)
        self.keep_rst = keep_rst
        self.debug = debug
        
        if config is None:
            self.config = {}
        else:
            self.config = config

        if self.debug:
            self.debug_dir = os.path.join(self.in_path,'debug')
            try: 
                os.makedirs(self.debug_dir)
            except FileExistsError:
                pass
        
        try:
            self.ipynb_path = glob_for_one_file(self.in_path, '*.ipynb')
        except RuntimeError:
            self.ipynb_path = None

    def nb_to_rst(self):
        """
        This converts the notebook found on init (at `self.ipynb_path`) to an
        rst file.
        """
        
        with io.open(self.ipynb_path, mode="r") as f:
            nb = nbformat.read(f, as_version=4)
       
        c = Config()
        c.update(self.config)

        rst_exporter = RSTExporter(config = c)
        nbconvert_writer = FilesWriter(build_directory=self.in_path)
        output, resources = rst_exporter.from_notebook_node(nb)
        nbconvert_writer.write(output, resources, notebook_name=self.paper_id)
        self.input_rst_file_path = glob_for_one_file(self.in_path, '*.rst') 

    def convert(self):
        """
        This executs the `nb_to_rst` conversion step.
        """
        if self.ipynb_path:
            print("Converting {0}.ipynb to {0}.rst".format(self.paper_id))
            self.nb_to_rst()
    
    def create_debug_file_path(self):
        rst_files = glob.glob(os.path.join(self.debug_dir, "*.rst"))
        import ipdb; ipdb.set_trace()
        self.num_debug_files = len(rst_files)
        file_basename = os.path.basename(self.input_rst_file_path)
        new_file_name = (os.path.splitext(file_basename)[0] +
                        str(self.num_debug_files + 1) + 
                        os.path.splitext(file_basename)[1])
        self.debug_file_path = os.path.join(self.debug_dir,new_file_name)
        

    def cleanup(self):
        """Applies various cleanup methods for rst converted from a notebook"""

        if self.ipynb_path and not self.keep_rst:
            
            if self.debug:
                self.create_debug_file_path()
                shutil.copy(self.input_rst_file_path, self.debug_file_path)

            os.remove(self.input_rst_file_path)
