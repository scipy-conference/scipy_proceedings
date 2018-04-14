#!/usr/bin/env python

import _mailer as mailer
from options import cfg2dict
from conf import proc_conf

args = mailer.parse_args()
config = cfg2dict('email.json')
scipy_config = cfg2dict(proc_conf)
config.update(scipy_config)

template = args.template or 'author-revision.txt'

for author in config['authors']:
    to = mailer.email_addr_from(author)
    config['committee'] =  mailer.create_committee(scipy_config)
    config['editor_email_string'] =  mailer.editor_email_string()
    config['recipients'] = to
    mailer.send_template(config['sender'], to, template, config)

print("Mail for %d authors." % len(config['authors']))
