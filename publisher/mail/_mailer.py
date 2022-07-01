import attr
from email.message import EmailMessage
import smtplib
import getpass
import time
import typing as t
import warnings

import sys

sys.path.insert(0, "..")
from build_template import _from_template


@attr.s(auto_attribs=True)
class Server:

    sender: str
    password: str
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    _session: t.Optional[smtplib.SMTP] = None

    def __enter__(self):
        self._session = self._get_session()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._session.quit()
        self._session = None

    def _get_session(self) -> smtplib.SMTP:
        session = smtplib.SMTP(self.smtp_server, self.smtp_port)

        session.ehlo()
        session.starttls()
        session.ehlo()
        session.login(self.sender, self.password)
        return session

    def send_message(self, message: EmailMessage, session=None) -> t.Dict:
        if session is None:
            session = self._session
        if session is not None:
            return session.send_message(message)
        else:
            raise TypeError("Must have active session to send mail")


@attr.s(auto_attribs=True)
class Mailer:

    template_path: str
    send: bool = False

    def send_all(self, template_data: t.List[t.Dict]):
        if self.send:
            sender = self.get_sender()
            password = self.get_password(sender)
            mailer = Server(sender=sender, password=password)
        else:
            sender = "test"

        emails = (self.make_email(sender, data) for data in template_data)
        if self.send:
            with mailer:
                for email in emails:
                    try:
                        errors = mailer.send_message(email)
                    except Exception as exc:
                        # if all addresses refuse the email, smtplib raises
                        warnings.warn(repr(exc))
                    else:
                        # if at least one to address succeeds, smtplib returns
                        # a dict of errors, potentially empty
                        if errors:
                            warnings.warn(repr(errors))
                        # GMail will close the connection if more than ~50 emails are sent
                        # per minute
                        # https://stackoverflow.com/questions/45756808/bulk-emails-failed-with-421-4-7-0-try-again-later
                        # https://support.google.com/a/answer/3726730?hl=en
                        time.sleep(2)

        else:
            # if we're doing a test run, dump everything to the terminal
            for email in emails:
                print("=" * 80)
                print(email.as_string())
                print("=" * 80)

    def make_email(self, sender: str, data: t.Dict) -> EmailMessage:
        email = EmailMessage()
        body: str = _from_template("../mail/templates/" + self.template_path, data)
        email.set_content(body)
        email["From"] = sender
        if "to_emails" in data:
            email["To"] = ", ".join([s for s in data["to_emails"] if s])
        if "cc_emails" in data:
            email["CC"] = ", ".join([s for s in data["cc_emails"] if s])
        if "subject" in data:
            email["Subject"] = data["subject"]
        return email

    @staticmethod
    def get_sender() -> str:
        return input("sender address: ").strip()

    @staticmethod
    def get_password(sender: str) -> str:
        return getpass.getpass(sender + "'s password:  ").strip()
