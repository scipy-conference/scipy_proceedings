#!/usr/bin/env python

import os

import _mailer as mailer
from conf import work_dir, toc_conf, proc_conf
import options

args = mailer.parse_args()
scipy_proc = options.cfg2dict(proc_conf)
toc = options.cfg2dict(toc_conf)

sender = scipy_proc['proceedings']['xref']['depositor_email']
template = 'doi-notification.txt'
template_data = scipy_proc.copy()
template_data['proceedings']['editor_email'] = ', '.join(template_data['proceedings']['editor_email'])

for paper in toc['toc']:

    template_data.update(paper)
    recipients = ','.join(template_data['author_email'])
    template_data['author'] = mailer.author_greeting(template_data['author'])
    template_data['author_email'] = ', '.join(template_data['author_email'])
    template_data['committee'] = '\n  '.join(template_data['proceedings']['editor'])

    mailer.send_template(sender, recipients, template, template_data)
