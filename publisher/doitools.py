#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""This module contains utility functions for generating DOIs and batch IDs
for submitting conference proceedings to CrossRef. The entry points of this
module are:

make_doi - for making an entire DOI, given as assigned prefix
make_batch_id - for making the identifier for submitting DOIs to CrossRef
"""

import re
from socket import gethostname
from subprocess import check_output
import time


class Clock:
    """Simple clock. Has one method which returns an integer and then stores
    that integer incremented by one"""


    def __init__(self):
        self._data = 0

    @property
    def clock(self):
        """Returns current integer and increments privately held integer"""
        result = self._data
        self._data += 1
        return result

def make_series_doi(prefix, issn):
    """Given prefix and issn, return appropriate doi for series"""
    formatted_issn = "issn.{}".format(issn)
    return '/'.join([prefix, formatted_issn])

def make_doi(prefix):
    """Given an assigned prefix, returns full, unique DOI for an object"""
    suffix = make_suffix()
    return '/'.join([prefix, suffix])

def make_suffix():
    """Returns DOI suffix for a paper composed of hostname, short commit
    hash, and clock int. This has moderate guarantees of uniqueness without
    maintaining state across sessions.
    """
    hostname = gethostname().split('.')[0]
    commit = get_commit()
    timestamp = get_clock()
    template = "{}-{}-{:03x}"
    return template.format(hostname, commit, timestamp)

def make_batch_id():
    """Returns moderately unique identifier to be used in the submission
    of a group of DOI metadata to CrossRef. For convenience, this uses the
    same logic as the suffix generator
    """
    hostname = gethostname().split('.')[0]
    commit = get_commit()
    timestamp = int(time.time() * 1000)
    template = "{}.{}-{:x}"
    return template.format(hostname, commit, timestamp)

def get_commit():
    """Returns short git commit hash. Must be called from within a valid
    repository.
    """
    result = check_output(
            ['git', 'rev-parse', '--verify', '--short', 'HEAD']
        ).decode('utf-8').replace('\n', '')
    return result

def get_clock():
    """Returns monotonically increasing integers.

    This function is a hacked implementation of the singleton pattern, and
    should be replaced by overwriting __new__ in the Clock class.
    """
    global _clock
    try:
        return _clock.clock
    except NameError:
        _clock = Clock()
        return _clock.clock
