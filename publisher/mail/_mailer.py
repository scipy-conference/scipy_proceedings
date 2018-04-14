import argparse
import smtplib
import os
import getpass
from email.mime.text import MIMEText

import sys
sys.path.insert(0, '..')

from conf import proc_conf
from options import cfg2dict
from build_template import _from_template


args = None
password = None


def author_greeting(names):
    if len(names) == 1:
        return names[0]
    else:
        return ', '.join(names[:-1]) + ', and ' + names[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Invite reviewers.")
    parser.add_argument('--send', action='store_true')
    parser.add_argument('--template', default=None)

    global args
    args = parser.parse_args()
    args.dry_run = not args.send

    if args.dry_run:
        print('*** This is a dry run.  Use --send to send emails.')

    return args


def load_config(conf_file):
    return cfg2dict(conf_file)


def get_password(sender):
    global password
    if not args.dry_run and not password:
        password = getpass.getpass(sender + "'s password:  ")


def email_addr_from(name_email):
    return '"%s" <%s>' % (name_email['name'], name_email['email'])


def create_committee(data=None):
    if data is None:
        data = cfg2dict(proc_conf)
    proc = data.get('proceedings', {})
    editors = proc.get('editor', [])
    # editor_email = [x.strip() for x in proc.get('editor_email', '').split(',')]
    editor_email = proc.get('editor_email')
    assert len(editors) == len(editor_email)
    return [{"name": name, 
             "email": email} 
            for name, email 
            in zip(editors, editor_email)]

def editor_email_string(data=None):
    if data is None:
        data = cfg2dict(proc_conf)
    editor_email_list = data.get('proceedings', {}).get('editor_email',[])
    return ', '.join(editor_email_list)

def create_message(recipient, template, template_data):
    template_data['email'] = recipient
    template_path = resolve_template(template, '../mail/templates/')
    return _from_template(template_path, template_data)


def resolve_template(template_name, base_dir):
    template_path = os.path.join(base_dir, template_name)
    # This needs to be add '.tmpl' because that's what _from_template expects
    if not os.path.exists(template_path +'.tmpl'):
        raise ValueError('There is no template file at {}.'.format(template_path+'.tmpl'))
    else:
        return os.path.abspath(template_path)


def send_template(sender, recipient, template, template_data,
                  smtp_server='smtp.gmail.com', smtp_port=587):
    if args.dry_run:
        print('Dry run -> not sending mail to %s' % recipient)
    else:
        get_password(sender['login'])
        print('-> %s' % recipient)
    
    message = create_message(recipient, template, template_data)

    if args.dry_run:
        print("=" * 80)
        print(message)
        print("=" * 80)
        return

    session = smtplib.SMTP(smtp_server, smtp_port)

    session.ehlo()
    session.starttls()
    session.ehlo
    session.login(sender['login'], password)

    session.sendmail(sender['name'], recipient, message)
    session.quit()
