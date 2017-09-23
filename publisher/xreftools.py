#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""This module contains the knowledge of how to take the metadata produced
during construction of the SciPy conference proceedings and functions for
generating valid DOIs and produces an xml file containing all of the necessary
metadata to submit DOIs to CrossRef. The XrefMeta class is the main entry
point, and depends on the individual conference information specified in
scipy_proceedings/scipy_proc.json, and the metadata for individual papers which
is contained in scipy_proceedings/publisher/toc.json.

Note: we currently implement the CrossRef "simple proceedings", where each
paper points to one proceedings. We are in the process of applying for an ISSN,
at which point XrefMeta will need to be updated to implement to "proceedings
series" schema.
"""

import lxml.etree as xml
from nameparser import HumanName
import time

from doitools import make_batch_id, make_doi

class XrefMeta:

    def __init__(self, scipy_conf, toc):
        self.scipy_entry = scipy_conf
        self.toc_entries = toc
        # lxml's implementation of xml location attributes is a little odd
        location = "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
        # this thing is the root node of allllll the elements to follow
        self.doi_batch = xml.Element('doi_batch',
            version="4.4.0",
            xmlns="http://www.crossref.org/schema/4.4.0",
            attrib={location: "http://www.crossref.org/schema/4.4.0 http://www.crossref.org/schemas/crossref4.4.0.xsd"}
        )

    def make_metadata(self, echo=False):
        """Build CrossRef compliant batch of DOI metadata, and store it
        internally.

        Meant to be called before 'write_metadata'. Set echo=True
        to send doi_batch to STDOUT
        """
        self.make_head()
        self.make_body()
        if echo:
            print(xml.dump(self.doi_batch))

    def make_head(self):
        """Build the metametadata, including timestamp and depositor email

        The depositor email is super important, because that's who gets
        contacted if the submission fails
        """
        head = xml.SubElement(self.doi_batch, 'head')
        doi_batch_id = xml.SubElement(head, 'doi_batch_id')
        doi_batch_id.text = make_batch_id()
        timestamp = xml.SubElement(head, 'timestamp')
        timestamp.text = str(time.mktime(time.gmtime()))
        depositor = xml.SubElement(head, 'depositor')
        depositor_name = xml.SubElement(depositor, 'depositor_name')
        depositor_name.text = self.scipy_entry["proceedings"]["xref"]["depositor_name"]
        depositor_email_address = xml.SubElement(depositor, 'email_address')
        depositor_email_address.text = self.scipy_entry["proceedings"]["xref"]["depositor_email"]
        registrant = xml.SubElement(head, 'registrant')
        registrant.text = self.scipy_entry["proceedings"]["xref"]["registrant"]

    def make_body(self):
        """Build the metadata, including the conference, proceedings, and
        individual papers
        """
        body = xml.SubElement(self.doi_batch, 'body')
        self.make_conference(body)

    def make_conference(self, body):
        """Build metadata for Scipy conference and individual papers"""
        conference = xml.SubElement(body, 'conference')
        event_metadata = xml.SubElement(conference, 'event_metadata')
        conference_name = xml.SubElement(event_metadata, 'conference_name')
        conference_name.text = self.scipy_entry['proceedings']['title']['conference']
        conference_acronym = xml.SubElement(event_metadata, 'conference_acronym')
        conference_acronym.text = self.scipy_entry['proceedings']['title']['acronym']
        conference_number = xml.SubElement(event_metadata, 'conference_number')
        conference_number.text = self.scipy_entry['proceedings']['title']['ordinal']
        conference_location = xml.SubElement(event_metadata, 'conference_location')
        conference_location.text = self.scipy_entry['proceedings']['location']
        conference_date = xml.SubElement(event_metadata, 'conference_date')
        conference_date.text = ' '.join([self.scipy_entry['proceedings']['dates'], self.scipy_entry['proceedings']['year']])
        self.make_conference_proceedings(conference)
        for entry in self.toc_entries:
            self.make_conference_papers(conference, entry)


    def make_conference_proceedings(self, conference):
        """Build metadata for the conference proceedings object

        'no_isbn' must be 'simple_series' or CrossRef will refuse DOIs.
        """
        proceedings_metadata = xml.SubElement(conference, 'proceedings_metadata')
        proceedings_title = xml.SubElement(proceedings_metadata, 'proceedings_title')
        proceedings_title.text = self.scipy_entry['proceedings']['title']['full']
        proceedings_subject = xml.SubElement(proceedings_metadata, 'proceedings_subject')
        proceedings_subject.text = "Scientific Computing with Python" # TODO: move to scipy_proc.json
        publisher = xml.SubElement(proceedings_metadata, 'publisher')
        publisher_name = xml.SubElement(publisher, 'publisher_name')
        publisher_name.text = 'SciPy' # TODO: move to scipy_proc.json
        publication_date = xml.SubElement(proceedings_metadata, 'publication_date')
        publication_year = xml.SubElement(publication_date, 'year')
        publication_year.text = self.scipy_entry['proceedings']['year']
        noisbn = xml.SubElement(proceedings_metadata, 'noisbn', reason="simple_series") # Do not modify, unless someone has actually gone and gotten us an ISBN
        proceedings_doi_data = xml.SubElement(proceedings_metadata, 'doi_data')
        proceedings_doi = xml.SubElement(proceedings_doi_data, 'doi')
        proceedings_doi.text = make_doi(self.scipy_entry["proceedings"]["xref"]["prefix"])
        proceedings_resource = xml.SubElement(proceedings_doi_data, 'resource')
        proceedings_resource.text = self.proceedings_url()

    def make_conference_papers(self, conference, entry):
        """Build metadata for all of the conference papers in a proceedings"""
        paper = xml.SubElement(conference, "conference_paper")
        paper_contributors = xml.SubElement(paper, 'contributors')
        for index, contributor in enumerate(entry.get('author', [])):
            # CrossRef has two kinds of authors: {'first', 'additional'}
            person_name = xml.SubElement(paper_contributors, 'person_name', contributor_role='author', sequence="additional" if index else "first") # first index value is 0
            first_name, last_name = split_name(contributor)
            given_name = xml.SubElement(person_name, 'given_name')
            given_name.text = first_name
            surname = xml.SubElement(person_name, 'surname')
            surname.text = last_name
        titles = xml.SubElement(paper, 'titles')
        title = xml.SubElement(titles, 'title')
        title.text = entry.get('title', '')
        publication_date = xml.SubElement(paper, 'publication_date', media_type="print")
        year = xml.SubElement(publication_date, 'year')
        year.text = self.scipy_entry['proceedings']['year']
        pages = xml.SubElement(paper, 'pages')
        first_page = xml.SubElement(pages, 'first_page')
        first_page.text = str(entry['page']['start'])
        last_page = xml.SubElement(pages, 'last_page')
        last_page.text = str(entry['page']['stop'])
        paper_doi_data = xml.SubElement(paper, 'doi_data')
        paper_doi = xml.SubElement(paper_doi_data, 'doi')
        paper_doi.text = make_doi(self.scipy_entry["proceedings"]["xref"]["prefix"])
        paper_resource = xml.SubElement(paper_doi_data, 'resource')
        paper_resource.text = self.paper_url(entry['paper_id'])

    def write_metadata(self, filepath):
        """Dump entire doi metadata batch to filepath"""
        xml.ElementTree(self.doi_batch).write(filepath)

    def paper_url(self, paper_id):
        """Return the url where a particular paper will end up.

        The "paper_id" is pulled out of the toc, and is literally whatever the
        author named the folder that they put their paper in
        """
        page = paper_id + '.html'
        return '/'.join([self.proceedings_url(), page])

    def proceedings_url(self):
        """Return the url for the entire proceedings.

        The rule for this is implicit and shared across a couple of the
        templating files, but appears to be the conference acronym plus
        the current year.
        """
        url_base = self.scipy_entry["proceedings"]["xref"]["resource_url"]
        title = self.scipy_entry["proceedings"]['title']['acronym'].lower()
        year = self.scipy_entry["proceedings"]['year']
        return  '/'.join([url_base, title+year])

def split_name(string, missing='MISSING'):
    """Splits human name into first and last components, as required by
    CrossRef schema rules.

    Names that cannot be parsed correctly will be replaced with 'MISSING' so
    as to be easier for human eyes to spot.
    """
    name = HumanName(string)
    first, last = name.first, name.last
    if not first:
        first = missing
    if not last:
        last = missing
    return first, last
