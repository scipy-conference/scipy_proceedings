from __future__ import unicode_literals, print_function

import os

import options

from testpath import tempdir

def test_dict2cfg():
    d = {'title': 'temp_title'}
    with tempdir.TemporaryDirectory() as td:
        loc_path = os.path.join(td, 'paper_stats.json')
        options.dict2cfg(d, loc_path)


def test_cfg2dict():
    d = {'title': 'temp_title'}
    with tempdir.TemporaryDirectory() as td:
        loc_path = os.path.join(td, 'paper_stats.json')
        options.dict2cfg(d, loc_path)
        test_d = options.cfg2dict(loc_path)
        assert test_d == d
