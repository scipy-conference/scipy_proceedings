#!/usr/bin/env python

import _mailer as mailer

args = mailer.parse_args()
config = mailer.load_config('email.json')

for author in config['authors']:
    to = mailer.email_addr_from(author)
    mailer.send_template(config['sender'], to, args.template, config)

print("Mail for %d authors." % len(config['authors']))
