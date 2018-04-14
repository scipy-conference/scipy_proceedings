#!/usr/bin/env python

from _mailer import Mailer, parse_args

from conf import toc_conf, proc_conf
from options import cfg2dict

args = parse_args()
template = args.template or 'call_for_papers.txt'

scipy_proc = cfg2dict(proc_conf)
toc = cfg2dict(toc_conf)

class AuthorMailer(Mailer):
    @property
    def recipients(self):
        return ', '.join(self.template_data['author_email'])

    @property
    def custom_data(self):
        return {'author': self.recipient_greeting(self.template_data['author'])}

for paper in toc['toc']:

    mailer = AuthorMailer(template=template, 
                          data_sources=[scipy_proc, 
                                        paper])
                                        
    recipients = ','.join(paper['author_email'])
    mailer.send_from_template(recipients)

