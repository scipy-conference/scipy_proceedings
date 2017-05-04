"""
Configuration utilities.
"""
from __future__ import print_function, unicode_literals

__all__ = ['options']

import os.path
import json
import io
import codecs

import conf
toc_conf   = conf.toc_conf
proc_conf  = conf.proc_conf

def get_config():
    config = cfg2dict(proc_conf)
    config.update(cfg2dict(toc_conf))
    return config

def cfg2dict(filename):
    """Return the content of a JSON config file as a dictionary.

    """
    if not os.path.exists(filename):
        print('*** Warning: %s does not exist.' % filename)
        return {}

    _backup_filename = filename+'.bak'
    if os.path.exists(_backup_filename):
        os.remove(_backup_filename)
        print('found previous backup file {}, removing...'.format(_backup_filename))

    try: 
        with io.open(filename,  mode='r', encoding='utf-8') as f:
            return json.loads(f.read())
    except ValueError as err:
        os.rename(filename,filename+'.bak')
        print('{} is not a valid json file, moving to {} for debugging.'
              'Running again will remove backup file.'
              .format(filename, _backup_filename))
        return {} 

def dict2cfg(d, filename):
    """Write dictionary out to config file.

    """
    with io.open(filename, mode='wb') as f:
        json.dump(d, codecs.getwriter('utf-8')(f), ensure_ascii=False)

def mkdir_p(dir):
    if os.path.isdir(dir):
        return
    os.makedirs(dir)

options = cfg2dict(proc_conf)
