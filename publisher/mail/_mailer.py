import argparse
import smtplib
import os
import getpass

from contextlib import contextmanager

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
    """
    Parameters
    ----------
    name_email: dict
        A dictionary of names and emails. 
        Expected fields: 
            'name': name of person
            'email': email of person
    """
    
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


def load_data_file(source_file):
        if os.path.exists(source_file):
            return cfg2dict(source_file)
        else:
             print('file at {} not found'.format(os.path.abspath(s)))


def load_data_sources(sources):
    """Combines dictionaries starting with the first included.
    """
    new_dict = {}
    for s in sources:
        if isinstance(s, dict):
            data = s
        elif isinstance(s, str):
            data = load_data_file(s)
        new_dict = {**new_dict, **data}
    return new_dict

class Mailer:
    
    def __init__(self,
                 sender=None,
                 template='',
                 base_dir='../mail/templates/',
                 data_sources=None,
                 smtp_server='smtp.gmail.com', 
                 smtp_port=587,
                 dry_run=True,
                 ):
        if sender is not None and isinstance(sender, dict):
            self.sender = sender
        
        
        if data_sources is None:
            data_sources = []
        elif isinstance(data_sources, (str, dict)):
            data_sources = [data_sources]
        elif isinstance(data_sources, list):
            data_sources = data_sources
        # we always need the email metadata, so let's make that default
        
        self.data_sources = ['./email.json'] + data_sources
        
        self.dry_run = True
        self.base_dir = base_dir
        self.template = template
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.aux_data = {}
        
    @staticmethod
    def recipient_greeting(names):
        if len(names) == 1:
            name_string = names[0]
        else:
            name_string = ', '.join(names[:-1]) + ', and ' + names[-1]
        return name_string
    
    @staticmethod
    def fancy_prep(data, name_key="names", email_key="emails"):
        names = data[name_key]
        emails = data[email_key]
        assert len(names) == len(emails)
        return [{"name": name, "email": email} 
                for name, email 
                in zip(names, emails)]
            
    @classmethod
    def fancy_emails(cls, data, name_key="names", email_key="emails"):
        """this is a method that takes a data object and gives back a email string
        
        First we preprocess the people with fancy_prep.
        
        Then we fileter to make sure all the emails and names are valid strings.
        We have to do the filtering because sometimes emails are nans.
        
        Then we join the list into a comma separated string with email appropriate formatting.
        
        Parameters:
        -----------
        data: dict
            this should have at least two fields, the vals of name_key and email_key
        name_key: str
            this is the key to indicate names
        email_key: str
            this is the key to indicate emails
        """
        people_gen = (p for p in cls.fancy_prep(data, name_key=name_key, email_key=email_key))
        are_str = lambda x: (isinstance(x['name'], str) and isinstance(x['email'], str)) 
        eml_strs = ('"{name}" <{email}>'.format(**p) for p in people_gen if are_str(p))
        return ", ".join(eml_strs)


    @property
    def recipients(self):
        return 'blah@blah.org'
        
    @property
    def sender(self):
        return self._sender if self._sender else self.template_data["sender"]
        
    @sender.setter
    def sender(self, value):
        if (isinstance(value, dict) and all(k in value for k in ('name','email'))):
            self._sender = value
        else:
            raise ValueError("You tried to set {} as the sender "
                             "but it has no 'name' and 'email' keys.".format(value))
        
    @property
    def password(self):
        if not self.dry_run and not self._password:
            self._password = getpass.getpass(sender + "'s password:  ")
        return self._password
    
    @property
    def template_data(self):
        return {**load_data_sources(self.data_sources), **self.aux_data}
        
    @property
    def template(self):
        return self._template if os.path.exists(self._template+'.tmpl') else ""
            
    @template.setter
    def template(self, value):
        self._template = resolve_template(value, self.base_dir)
        
    def prep_data(self, data=None):
        data = data if (data and isinstance(data, dict)) else {}
        self.aux_data = {**self.common_data, **self.custom_data, **data}
    
    @property
    def common_data(self):
        return {'editor_email_string':  editor_email_string(),
                'committee': create_committee()}
    
    @property
    def custom_data(self):
        return {}
            
    def send_from_template(self, recipients=None, data=None):
        """
        
        Parameters
        ----------
        recipient: str
            email 
        
        """
        data = {} if data is None else data
        recipients = self.recipients if recipients is None else recipients
        self.prep_data({**data, 'recipients':recipients})

        message = _from_template(self.template, self.template_data)
        
        if self.dry_run:
            self.display_message(recipients, message)
        else:
            self.send_mail(recipients, message)


    def display_message(self, recipients, message):
        print('Dry run -> not sending mail to %s' % recipients)
        print("=" * 80)
        print(message)
        print("=" * 80)
    
    def send_mail(self, recipients, message):
        with self.session() as session:
            session.sendmail(self.sender['name'], recipients, message)

    @contextmanager
    def session(self):
        
        self.get_password(self.sender['login'])
        print('-> %s' % self.recipients)
        session = smtplib.SMTP(self.smtp_server, self.smtp_port)

        session.ehlo()
        session.starttls()
        session.ehlo
        session.login(self.sender['login'], self.password)
        yield session

        session.quit()
