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

from doitools import make_batch_id

class XrefMeta:

    def __init__(self, scipy_conf, toc, slides):
        self.scipy_entry = scipy_conf
        self.toc_entries = toc
        self.slide_entries = slides

    def make_metadata(self, echo=False):
        """Build CrossRef compliant batch of DOI metadata, and store it
        internally.

        Meant to be called before 'write_metadata'. Set echo=True
        to send doi_batch to STDOUT
        """
        self.papers_batch = self.make_papers_batch()
        self.slides_batch = self.make_slides_batch()
        if echo:
            print(xml.dump(self.papers_batch))
            print(xml.dump(self.slides_batch))

    def make_batch(self):
        """Build the metametadata, including timestamp and depositor email

        The depositor email is super important, because that's who gets
        contacted if the submission fails
        """
        # lxml's implementation of xml location attributes is a little odd
        location = "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
        batch = xml.Element('doi_batch',
            version="4.4.2",
            xmlns="http://www.crossref.org/schema/4.4.2",
            attrib={location: "http://www.crossref.org/schema/4.4.2 http://www.crossref.org/schemas/crossref4.4.2.xsd"}
        )
        head = xml.SubElement(batch, 'head')
        doi_batch_id = xml.SubElement(head, 'doi_batch_id')
        doi_batch_id.text = make_batch_id()
        timestamp = xml.SubElement(head, 'timestamp')
        timestamp.text = str(int(time.mktime(time.gmtime())))
        depositor = xml.SubElement(head, 'depositor')
        depositor_name = xml.SubElement(depositor, 'depositor_name')
        depositor_name.text = self.scipy_entry["proceedings"]["xref"]["depositor_name"]
        depositor_email_address = xml.SubElement(depositor, 'email_address')
        depositor_email_address.text = self.scipy_entry["proceedings"]["xref"]["depositor_email"]
        registrant = xml.SubElement(head, 'registrant')
        registrant.text = self.scipy_entry["proceedings"]["xref"]["registrant"]
        return batch

    def make_papers_batch(self):
        """Build the metadata for conference papers
        """
        batch = self.make_batch()
        body = xml.SubElement(batch, 'body')
        self.make_conference(body)
        return batch

    def make_slides_batch(self):
        """Build the metadata for conference slides
        """
        batch = self.make_batch()
        body = xml.SubElement(batch, 'body')
        self.make_database(body)
        return batch

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
        proceedings_series_metadata = xml.SubElement(conference, 'proceedings_series_metadata')
        series_metadata = xml.SubElement(proceedings_series_metadata, 'series_metadata')
        titles = xml.SubElement(series_metadata, 'titles')
        title = xml.SubElement(titles, 'title')
        title.text = self.scipy_entry['series']['title']['full']
        original_language_title = xml.SubElement(titles, 'original_language_title')
        original_language_title.text = self.scipy_entry['series']['title']['full']
        issn = xml.SubElement(series_metadata, 'issn')
        issn.text = self.scipy_entry['series']['xref']['issn']
        # TODO: decide on whether/how to archive proceedings
        # series_archive_locations = xml.SubElement(series_metadata, 'archive_locations')
        # series_archive_location = xml.SubElement(series_archive_locations, 'archive_location')
        # series_archive_location.text = "Internet Archive"
        series_doi_data = xml.SubElement(series_metadata, 'doi_data')
        series_doi = xml.SubElement(series_doi_data, 'doi')
        series_doi.text = self.scipy_entry['series']['doi']
        series_resource = xml.SubElement(series_doi_data, 'resource')
        series_resource.text = self.scipy_entry["series"]["xref"]["resource_url"]
        proceedings_title = xml.SubElement(proceedings_series_metadata, 'proceedings_title')
        proceedings_title.text = self.scipy_entry['proceedings']['title']['full']
        proceedings_subject = xml.SubElement(proceedings_series_metadata, 'proceedings_subject')
        proceedings_subject.text = "Scientific Computing with Python" # TODO: move to scipy_proc.json
        publisher = xml.SubElement(proceedings_series_metadata, 'publisher')
        publisher_name = xml.SubElement(publisher, 'publisher_name')
        publisher_name.text = 'SciPy' # TODO: move to scipy_proc.json
        publication_date = xml.SubElement(proceedings_series_metadata, 'publication_date')
        publication_year = xml.SubElement(publication_date, 'year')
        publication_year.text = self.scipy_entry['proceedings']['year']
        noisbn = xml.SubElement(proceedings_series_metadata, 'noisbn', reason="simple_series") # Do not modify, unless someone has actually gone and gotten us an ISBN
        proceedings_doi_data = xml.SubElement(proceedings_series_metadata, 'doi_data')
        proceedings_doi = xml.SubElement(proceedings_doi_data, 'doi')
        proceedings_doi.text = self.scipy_entry["proceedings"]["doi"]
        proceedings_resource = xml.SubElement(proceedings_doi_data, 'resource')
        proceedings_resource.text = self.proceedings_url()

    def make_conference_papers(self, conference, entry):
        """Build metadata for all of the conference papers in a proceedings"""
        paper = xml.SubElement(conference, "conference_paper")
        paper_contributors = xml.SubElement(paper, 'contributors')
        orcid_map = entry.get('author_orcid_map', {})
        for index, contributor in enumerate(entry.get('author', [])):
            # CrossRef has two kinds of authors: {'first', 'additional'}
            person_name = xml.SubElement(paper_contributors, 'person_name', contributor_role='author', sequence="additional" if index else "first") # first index value is 0
            first_name, last_name = split_name(contributor)
            given_name = xml.SubElement(person_name, 'given_name')
            given_name.text = first_name
            surname = xml.SubElement(person_name, 'surname')
            surname.text = last_name
            if contributor in orcid_map:
                orcid = xml.SubElement(person_name, 'ORCID')
                orcid.text = 'https://orcid.org/' + orcid_map[contributor]
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
        paper_doi.text = entry['doi']
        paper_resource = xml.SubElement(paper_doi_data, 'resource')
        paper_resource.text = self.paper_url(entry['paper_id'])

    def make_database(self, body):
        """Build metadata for the location, or database, where we are putting
        presentations
        """
        database = xml.SubElement(body, 'database')
        database_metadata = xml.SubElement(database, 'database_metadata', language='en')
        contributors = xml.SubElement(database_metadata, 'contributors')
        person_name = xml.SubElement(contributors, 'person_name', contributor_role='editor', sequence='first')
        first_name, last_name = split_name(self.scipy_entry['proceedings']['xref']['depositor_name'])
        given_name = xml.SubElement(person_name, 'given_name')
        given_name.text = first_name
        surname = xml.SubElement(person_name, 'surname')
        surname.text = last_name
        titles = xml.SubElement(database_metadata, 'titles')
        title = xml.SubElement(titles, 'title')
        title.text = self.scipy_entry['series']['title']['full']
        for group, entries in self.slide_entries.items():
            for entry in entries:
                entry['group'] = group
                self.make_dataset(database, entry)

    def make_dataset(self, database, entry):
        """Build metadata for a single presentation
        """
        dataset = xml.SubElement(database, "dataset", dataset_type='other')
        dataset_contributors = xml.SubElement(dataset, 'contributors')
        for index, contributor in enumerate(entry.get('authors', [])):
            # CrossRef has two kinds of authors: {'first', 'additional'}
            person_name = xml.SubElement(dataset_contributors, 'person_name', contributor_role='author', sequence="additional" if index else "first") # first index value is 0
            # author entries for presentations are {'name', 'affiliation', 'orcid'}
            full_name = contributor["name"]
            first_name, last_name = split_name(full_name)
            given_name = xml.SubElement(person_name, 'given_name')
            given_name.text = first_name
            surname = xml.SubElement(person_name, 'surname')
            surname.text = last_name
            orcid = contributor.get('orcid')
            if orcid:
                orcid_entry = xml.SubElement(person_name, 'ORCID')
                orcid_entry.text = 'https://orcid.org/' + orcid
        titles = xml.SubElement(dataset, 'titles')
        title = xml.SubElement(titles, 'title')
        title.text = entry.get('title', '')
        database_date = xml.SubElement(dataset, 'database_date')
        publication_date = xml.SubElement(database_date, 'publication_date')
        year = xml.SubElement(publication_date, 'year')
        year.text = self.scipy_entry['proceedings']['year']
        description = xml.SubElement(dataset, "description")
        description.text = entry.get('description', 'Material presented at SciPy')
        program = xml.SubElement(dataset, 'program', name='relations', xmlns= "http://www.crossref.org/relations.xsd")
        related_item = xml.SubElement(program, 'related_item')
        inter_work_relation = xml.SubElement(related_item, 'inter_work_relation', attrib={'relationship-type': 'isPartOf', 'identifier-type': 'doi'})
        inter_work_relation.text = self.scipy_entry['series']['doi']
        doi_data = xml.SubElement(dataset, 'doi_data')
        doi = xml.SubElement(doi_data, 'doi')
        doi.text = entry['doi']
        resource = xml.SubElement(doi_data, 'resource')
        resource.text = entry.get('zenodo_url', 'https://fake-url.place')

    def write_metadata(self, filepath_root):
        """Dump doi metadata batches to filepath_root + suffix"""
        xml.ElementTree(self.papers_batch).write(filepath_root + '_papers.xml')
        xml.ElementTree(self.slides_batch).write(filepath_root + '_slides.xml')

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
