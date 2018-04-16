#!/usr/bin/env python

from _mailer import Mailer, parse_args

from options import cfg2dict



class AuthorMailer(Mailer):
    @property
    def recipients_list(self):
        send_to = {"names": ["David Lippa", "M Pacer", "Elizabeth Seiver"], 
                   "emails": ["dalippa@cs.brandeis.edu", "mpacer@berkeley.edu", "elizabethseiver@gmail.com"]}
        # send_to = {"names": ["David Lippa"],
        #            "emails": ["dalippa@cs.brandeis.edu"]}
        return self.fancy_email_list(send_to, name_key="names", email_key="emails")
        
        # send_to = self.template_data
        # return self.fancy_emails(send_to, name_key="authors", email_key="emails")

    @property
    def custom_data(self):
        return {'author': self.recipient_greeting(self.template_data['authors'])}


args = parse_args()
template = args.template or 'call_for_papers.txt'

accepts = cfg2dict('./accepted_talks_and_posters.json')
for proposal, info in accepts.items():
    mailer = AuthorMailer(template=template, 
                          data_sources=[info],
                          dry_run=args.dry_run)
    
    mailer.send_from_template()
    import ipdb; ipdb.set_trace()
