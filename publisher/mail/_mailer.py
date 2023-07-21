import attr
import base64
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
import os
import sys
import time
import typing as t

sys.path.insert(0, "..")
from build_template import _from_template


@attr.s(auto_attribs=True)
class Server:

    _creds: Credentials = None
    _client: Resource = None
    # the scopes need to match what permissions we have granted to this
    # application in GCP
    _scopes: t.List[str] = ['https://www.googleapis.com/auth/gmail.send']
    _google_app_creds_var: str = 'GOOGLE_APPLICATION_CREDENTIALS'

    def __enter__(self):
        self._client = self._get_client()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._client.close()
        self._client = None

    def _get_credentials(self) -> Credentials:
        if self._creds is None:
            # the auth process needs our application token and secret
            # this is in a json blob that you download from GCP,
            if self._google_app_creds_var not in os.environ:
                raise RuntimeError(f"App creds location missing from {self._google_app_creds_var} env var")
            service_creds_file = os.environ[self._google_app_creds_var]
            # this opens a tab on the host's default browser and asks
            # the user to give our app permission to send emails inside
            # their account
            flow = InstalledAppFlow.from_client_secrets_file(service_creds_file, self._scopes)
            creds = flow.run_local_server()
            self._creds = creds
        return self._creds

    def _get_client(self) -> Resource:
        if self._client is None:
            creds = self._get_credentials()
            # this returns our gmail client object
            client = build('gmail', 'v1', credentials=creds)
            self._client = client
        return self._client

    def _convert_email_format(self, message: EmailMessage) -> t.Dict[str, str]:
        # the google api docs say this format should be
        # {'message': {'raw': <base64 encoded email object>}}
        # but this fails if you have the outer 'message' field
        payload = {
            'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()
        }
        return payload

    def send_message(self, message: EmailMessage) -> t.Dict[str, t.Any]:
        payload = self._convert_email_format(message)
        client = self._get_client()
        request = client.users().messages().send(userId="me", body=payload)
        # client actions don't execute immediately, for reasons
        return request.execute()


@attr.s(auto_attribs=True)
class Mailer:

    template_path: str
    send: bool = False
    sender: str = 'scipy.proceedings@gmail.com'

    def send_all(self, template_data: t.List[t.Dict]):
        emails = (self.make_email(self.sender, data) for data in template_data)
        if self.send:
            # if we're doing this for real, make our mail client and call its
            # send method for each email we have to send
            mailer = Server()
            with mailer:
                for email in emails:
                    try:
                        response = mailer.send_message(email)
                    except HttpError as exc:
                        # if something goes wrong, google throws a generic error
                        # type, but the message will have the http status code
                        # and reason in it
                        print(repr(exc))
                        # dump the email so we can see which ones didn't send
                        print(email.as_string())

                    else:
                        # on success, the client returns a message and thread id
                        print(response)
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
