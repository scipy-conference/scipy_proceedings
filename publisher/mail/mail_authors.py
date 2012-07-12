#!/usr/bin/env python

import _mailer as mailer

args = mailer.parse_args()
config = mailer.load_config('email.json')

for author in config['authors']:
    mailer.send_template(config['sender'], author, args.template, config)
