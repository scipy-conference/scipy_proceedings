import argparse
import smtplib
import os
from email.mime.text import MIMEText

import sys
sys.path.insert(0, '..')
from conf import work_dir
from options import cfg2dict
from build_template import _from_template


args = None
password = None


def parse_args():
    parser = argparse.ArgumentParser(description="Invite reviewers.")
    parser.add_argument('--dry-run', default=True)
    parser.add_argument('--template', default=None)

    global args
    args = parser.parse_args()

    if args.dry_run:
        print '*** This is a dry run.  Use --dry-run=False to send emails.'

    return args


def load_config(conf_file):
    return cfg2dict(conf_file)


def get_password():
    global password
    if not args.dry_run and not password:
        password = getpass.getpass(config['sender']['login']+"'s password:  ")


def send_template(sender, recipient, template, template_data,
                  smtp_server='smtp.gmail.com', smtp_port=587):
    if args.dry_run:
        print 'Dry run -> not sending mail to %s' % recipient
    else:
        get_password()
        print '-> %s' % recipient

    template_data['email'] = recipient
    message = _from_template('../mail/templates/' + template, template_data)

    if args.dry_run:
        print "=" * 80
        print message
        print "=" * 80

        return

    session = smtplib.SMTP(smtp_server, smtp_port)

    session.ehlo()
    session.starttls()
    session.ehlo
    session.login(sender['login'], password)

    session.sendmail(sender['name'], recipient, message)
    session.quit()
