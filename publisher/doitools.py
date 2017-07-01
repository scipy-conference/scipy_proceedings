#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
from socket import gethostname
from subprocess import check_output
import time


class Clock:

    def __init__(self):
        self._data = 0

    @property
    def clock(self):
        result = self._data
        self._data += 1
        return result


def make_doi(prefix):
    suffix = make_suffix()
    return '/'.join([prefix, suffix])

def make_suffix():
    hostname = gethostname().split('.')[0]
    commit = get_commit()
    timestamp = get_clock()
    template = "{}-{}-{:03x}"
    return template.format(hostname, commit, timestamp)

def make_batch_id():
    hostname = gethostname().split('.')[0]
    commit = get_commit()
    timestamp = int(time.time() * 1000)
    template = "{}.{}-{:x}"
    return template.format(hostname, commit, timestamp)

def get_commit():
    result = check_output(
            ['git', 'rev-parse', '--verify', '--short', 'HEAD']
        ).decode('utf-8').replace('\n', '')
    return result

def get_clock():
    global _clock
    try:
        return _clock.clock
    except NameError:
        _clock = Clock()
        return _clock.clock
