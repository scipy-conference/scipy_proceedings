#!/usr/bin/env python

import _mailer as mailer
from conf import toc_conf, proc_conf
from options import cfg2dict

args = mailer.parse_args()
scipy_proc = cfg2dict(proc_conf)
toc = cfg2dict(toc_conf)

sender = scipy_proc['proceedings']['xref']['depositor_email']
template = 'doi-notification.txt'
template_data = scipy_proc.copy()

for paper in toc['toc']:

    template_data.update(paper)
    recipients = ','.join(template_data['author_email'])
    template_data['author'] = mailer.author_greeting(template_data['author'])
    template_data['committee'] = mailer.create_committee()
    template_data['editor_email_string'] =  mailer.editor_email_string()
    template_data['recipients'] = ', '.join(template_data['author_email'])

    mailer.send_template(sender, recipients, template, template_data)
