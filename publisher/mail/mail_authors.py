#!/usr/bin/env python

from copy import deepcopy

import pandas as pd

from _mailer import Mailer

# that import will have added our parent directory to the import path
# TODO fix the import structure of publish
from conf import proc_conf
import options


def main(data_path: str, send: bool = False):
    scipy_proc = options.cfg2dict(proc_conf)
    proceedings = scipy_proc["proceedings"]

    template = "author-invitation.txt"

    # we don't know what the author dump will look like yet, so we may need
    # to tweak this later
    # current assumption is that we will have table with fields for the
    # author's name, email address, and the paper title
    data = pd.read_csv(data_path)

    email_data = []
    for label, row in data.iterrows():
        one_email = deepcopy(scipy_proc)
        one_email["cc_emails"] = one_email["proceedings"]["editor_email"]
        one_email["to_name"] = row["name"]
        one_email["to_emails"] = [row["email"]]
        one_email["title"] = row["title"]
        one_email["subject"] = f"Invitation to submit a full paper for {proceedings['title']['acronym']} {proceedings['year']}"
        email_data.append(one_email)

    mailer = Mailer(template, send)
    mailer.send_all(email_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")
    parser.add_argument("data_path", help="path to csv of author data")

    args = parser.parse_args()

    if not args.send:
        print("*** This is a dry run.  Use --send to send emails.")

    main(**vars(args))
