#!/usr/bin/python

import getpass
import os

import gmail
from conf import work_dir
from options import cfg2dict
from build_template import _from_template

email_conf = os.path.join(work_dir, 'email.json')
config = cfg2dict(email_conf)

for reviewer_info in config['reviewers']:
    for p in reviewer_info['papers']:
        if not os.path.isdir(os.path.join(work_dir, '../papers/', p)):
            raise RuntimeError("Paper %s not found..refusing to generate emails." % p)

password = getpass.getpass(config['sender']['login']+"'s password:  ")

for reviewer_info in config['reviewers']:
    reviewer_config = config.copy()
    reviewer_config.update(reviewer_info)
    reviewer = reviewer_info['email']
    print "Sending invite to " + reviewer

    msg = _from_template('reviewer-invite.txt', reviewer_config)
    gmail.sendmail(config['sender'], reviewer, msg, password)
