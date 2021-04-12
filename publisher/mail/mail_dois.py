#!/usr/bin/env python

from copy import deepcopy
import typing as t

from _mailer import Mailer

# that import will have added our parent directory to the import path
# TODO fix the import structure of publish
from conf import toc_conf, proc_conf
import options


def main(send: bool = False):

    scipy_proc = options.cfg2dict(proc_conf)
    proceedings = scipy_proc["proceedings"]
    # the doi data is importable via the config module
    toc = options.cfg2dict(toc_conf)

    template = "doi-notification.txt"

    email_data = []
    for paper in toc["toc"]:
        one_email = deepcopy(scipy_proc)
        one_email.update(paper)
        one_email["cc_emails"] = one_email["proceedings"]["editor_email"]
        one_email["to_name"] = author_greeting(one_email["author"])
        one_email["to_emails"] = paper["author_email"]
        one_email["subject"] = f"Your {proceedings['title']['acronym']} {proceedings['year']} Paper DOI"
        email_data.append(one_email)
    mailer = Mailer(template, send)
    mailer.send_all(email_data)


def author_greeting(names: t.List[str]) -> str:
    if len(names) == 1:
        return names[0]
    else:
        return ", ".join(names[:-1]) + ", and " + names[-1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")

    args = parser.parse_args()

    if not args.send:
        print("*** This is a dry run.  Use --send to send emails.")

    main(**vars(args))
