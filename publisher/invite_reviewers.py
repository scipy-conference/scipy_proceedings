#!/usr/bin/env python

import getpass
import os
import sys

import gmail
from conf import work_dir
from options import cfg2dict
from build_template import _from_template

dry_run = (len(sys.argv) < 2) or (sys.argv[1] != '--send-emails')

email_conf = os.path.join(work_dir, 'email.json')
config = cfg2dict(email_conf)

for reviewer_info in config['reviewers']:
    for p in reviewer_info['papers']:
        if not os.path.isdir(os.path.join(work_dir, '../papers/', p)):
            raise RuntimeError("Paper %s not found..refusing to generate emails." % p)

if not dry_run:
    password = getpass.getpass(config['sender']['login']+"'s password:  ")

for reviewer_info in config['reviewers']:
    reviewer_config = config.copy()
    reviewer_config.update(reviewer_info)
    reviewer = reviewer_info['email']
    print "Sending invite to " + reviewer

    msg = _from_template('reviewer-invite.txt', reviewer_config)
    if dry_run:
        print "=" * 78
        print msg
        print "=" * 78
    else:
        gmail.sendmail(config['sender'], reviewer, msg, password)

paper_reviewers = {}
for reviewer_info in config['reviewers']:
    for paper in reviewer_info['papers']:
        d = paper_reviewers.setdefault(paper, [])
        d.append(reviewer_info['name'])

for paper in paper_reviewers:
    print "%s:" % paper
    for reviewer in paper_reviewers[paper]:
        print "->", reviewer
    print

print "Papers:", len(paper_reviewers)
print "Reviewers:", len(config['reviewers'])
print

if dry_run:
    print "** This was a dry run.  If all looks good, send the invitations"
    print "** using ./invite_reviewers.py --send-mail"
