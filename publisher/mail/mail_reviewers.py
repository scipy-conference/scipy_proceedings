#!/usr/bin/env python

import _mailer as mailer
import os
from conf import work_dir

args = mailer.parse_args()
config = mailer.load_config('email.json')


for reviewer_info in config['reviewers']:
    for p in reviewer_info['papers']:
        if not os.path.isdir(os.path.join(work_dir, '../papers/', p)):
            raise RuntimeError("Paper %s not found..refusing to generate emails." % p)


for reviewer_info in config['reviewers']:
    reviewer_config = config.copy()
    reviewer_config.update(reviewer_info)
    reviewer = reviewer_info['email']

    to = mailer.email_addr_from(reviewer_info)
    mailer.send_template(config['sender'], to + ', ' + config['cced'],
                         'reviewer-invite.txt', reviewer_config)


# Generate a summary of emails sent

paper_reviewers = {}
for reviewer_info in config['reviewers']:
    for paper in reviewer_info['papers']:
        d = paper_reviewers.setdefault(paper, [])
        d.append(reviewer_info['name'])

for paper in paper_reviewers:
    print("%s:" % paper)
    for reviewer in paper_reviewers[paper]:
        print("->", reviewer)
    print

print("Papers:", len(paper_reviewers))
print("Reviewers:", len(config['reviewers']))
print
