#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import lxml.etree as xml
from nameparser import HumanName
import time

from doitools import make_batch_id, make_doi

class XrefMeta:

    def __init__(self, scipy_conf, toc):
        self.scipy_entry = scipy_conf
        self.toc_entries = toc

        location = "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
        self.doi_batch = xml.Element('doi_batch',
            version="4.4.0",
            xmlns="http://www.crossref.org/schema/4.4.0",
            attrib={location: "http://www.crossref.org/schema/4.4.0 http://www.crossref.org/schemas/crossref4.4.0.xsd"}
        )

    def make_metadata(self, echo=False):
        self.make_head()
        self.make_body()
        if echo:
            print(xml.dump(self.doi_batch))

    def make_head(self):
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
        body = xml.SubElement(self.doi_batch, 'body')
        self.make_conference(body)

    def make_conference(self, body):
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
        proceedings_metadata = xml.SubElement(conference, 'proceedings_metadata')
        proceedings_title = xml.SubElement(proceedings_metadata, 'proceedings_title')
        proceedings_title.text = self.scipy_entry['proceedings']['title']['full']
        proceedings_subject = xml.SubElement(proceedings_metadata, 'proceedings_subject')
        proceedings_subject.text = "Scientific Computing with Python"
        publisher = xml.SubElement(proceedings_metadata, 'publisher')
        publisher_name = xml.SubElement(publisher, 'publisher_name')
        publisher_name.text = 'SciPy'
        publication_date = xml.SubElement(proceedings_metadata, 'publication_date')
        publication_year = xml.SubElement(publication_date, 'year')
        publication_year.text = self.scipy_entry['proceedings']['year']
        noisbn = xml.SubElement(proceedings_metadata, 'noisbn', reason="simple_series")
        proceedings_doi_data = xml.SubElement(proceedings_metadata, 'doi_data')
        proceedings_doi = xml.SubElement(proceedings_doi_data, 'doi')
        proceedings_doi.text = make_doi(self.scipy_entry["proceedings"]["xref"]["prefix"])
        proceedings_resource = xml.SubElement(proceedings_doi_data, 'resource')
        proceedings_resource.text = self.proceedings_url()

    def make_conference_papers(self, conference, entry):
        paper = xml.SubElement(conference, "conference_paper")
        paper_contributors = xml.SubElement(paper, 'contributors')
        for index, contributor in enumerate(entry.get('author', [])):
            person_name = xml.SubElement(paper_contributors, 'person_name', contributor_role='author', sequence="additional" if index else "first")
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
        xml.ElementTree(self.doi_batch).write(filepath)

    def paper_url(self, paper_id):
        page = paper_id + '.htm'
        return '/'.join([self.proceedings_url(), page])

    def proceedings_url(self):
        url_base = self.scipy_entry["proceedings"]["xref"]["resource_url"]
        title = self.scipy_entry["proceedings"]['title']['acronym'].lower()
        year = self.scipy_entry["proceedings"]['year']
        return  '/'.join([url_base, title+year])

def split_name(string, missing='MISSING'):
    name = HumanName(string)
    first, last = name.first, name.last
    if not first:
        first = missing
    if not last:
        last = missing
    return first, last
