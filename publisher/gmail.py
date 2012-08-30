#!/usr/bin/python
import smtplib
from email.mime.text import MIMEText

def sendmail(sender, recipient, msg, passwd, SMTP_SERVER = 'smtp.gmail.com', SMTP_PORT = 587):
    session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
     
    session.ehlo()
    session.starttls()
    session.ehlo
    session.login(sender['login'], passwd)

    session.sendmail(sender['name'], recipient, msg)
    session.quit()
