#!/usr/bin/env python

from copy import deepcopy
import typing as t

import pandas as pd

from _mailer import Mailer

# that import will have added our parent directory to the import path
# TODO fix the import structure
from conf import proc_conf
import options


def main(data_path: str, send: bool = False):
    scipy_proc = options.cfg2dict(proc_conf)
    proceedings = scipy_proc["proceedings"]

    template = "reviewer-reminder.txt"

    # we make this table by hand in google sheets when the committee divvies
    # up reviewers among papers -- the format we have been using is something like
    # | paper title | PR url | reviewer one | reviewer two | ... | reviewer n |
    # where each reviewer field is like "firstname lastname <email@provider>"
    data = pd.read_csv(data_path)
    data = pd.melt(
        data, id_vars=["title", "url"], var_name="number", value_name="reviewer"
    )
    data = data.dropna()

    email_data = []
    for label, row in data.iterrows():
        one_email = deepcopy(scipy_proc)
        one_email["cc_emails"] = one_email["proceedings"]["editor_email"]
        name, email = split_email_field(row["reviewer"])
        if "@" in email:
            # sometimes we won't have reviewer emails, so that field might be
            # firstname lastname and nothing else
            one_email["to_emails"] = [email]
            one_email["to_name"] = name
            one_email["title"] = row["title"].replace("Paper: ", "")
            one_email["url"] = row["url"]
            one_email["subject"] = f"{proceedings['title']['acronym']} {proceedings['year']} invitation to review proceedings"
            email_data.append(one_email)
        else:
            print(f"could not parse {row['reviewer']}")

    mailer = Mailer(template, send)
    mailer.send_all(email_data)


def split_email_field(field: str) -> t.Tuple[str, str]:
    if "<" in field:
        name, email = field.split("<")
        name = name.strip()
        email = email.replace(">", "").strip()
    else:
        name = field.strip()
        email = field.strip()
    return name, email


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")
    parser.add_argument("data_path")

    args = parser.parse_args()

    if not args.send:
        print("*** This is a dry run.  Use --send to send emails.")

    main(**vars(args))
