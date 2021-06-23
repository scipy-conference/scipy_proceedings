
:author: Haw-minn Lu
:email: hlu@westhealth.org
:institution: Gary and Mary West Health Institute

:author: José Unpingco
:email: jhunpingco@westhealth.org
:institution: Gary and Mary West Health Institute

:bibliography: ourbib

=============================================================================
How PDFrw and fillable forms improves throughput at a Covid-19 Vaccine Clinic
=============================================================================

.. class:: abstract

PDFrw was used to prepopulate Covid-19 vaccination forms to improve the efficiency and integrity of the vaccination process in terms of federal and state privacy requirements.  We will describe the vaccination process from the initial appointment, through the vaccination delivery, to the creation of subsequent required documentation. Although Python modules for PDF generation are common, they struggle with managing fillable forms where a fillable field may appear multiple times within the same form.  Additionally, field types such as checkboxes, radio buttons, lists and combo boxes are not straightforward to programmatically fill. Another challenge is combining multiple *filled* forms while maintaining the integrity of the values of the fillable fields.  Additionally, HIPAA compliance issues are discussed.

.. class:: keywords

   machine learning, classification, categorical encoding

Introduction
------------

The coronavirus pandemic has been perhaps one of the most disruptive nationwide
events in living memory. The frail, vulnerable, and elderly are
disproportionately affected by serious hospitalizations and deaths.  While the
near miraculous pace of development of effective vaccines was potential
salvation from the situation, the logistical challenges are immense
particularly, when it comes to the elderly.

When vaccination centers and clinics began to be established, all required
appointments and nearly all appointments had to be made online. Providing
vaccines to the most vulnerable population especially in the early stages of
the vaccine rollouts proved challenging as seniors are less likely to be tech
saavy and have transportation challenges.

As a personal anecdote, when vaccinations were open to all people 65 and older,
the author ventured to get my parents vaccinated. The process required periodic pinging of the 
appointment site for a vaccine supercenter. This process took a week of constant trying until an appointment could be made. Beyond perisistence, it required observing the pattern of when batches of new appointments were made available. Needless to say seniors even those with some amount of technological saavy  would have had extreme difficulty.

To address this the Gary and Mary West PACE (WestPACE) center established a
pop-up point of distribution (POD) for the COVID-19 vaccine :cite:`press`
specifically for the elderly with emphasis on those who are most vulnerable.
The success in the POD was touted in the local news media :cite:`knsd`
:cite:`kpbs` and caught the attention of the State of California who asked WestPACE's sister
organization the Gary and Mary West Health Institute to develop a playbook for
the deploying a pop-up POD :cite:`pod`.

This paper gives a little more background of the effort. Next the overall
infrastructure and information flow is descried. Finally, a very detailed
discussion on the use of python and the :code:`PDFrw` library to address a
major bottleneck and volunteer pain point.

Background
----------

WestPACE operates a Program of All-Inclusive Care for the Elderly (PACE) center
which provides nursing-home-level care. By the nature of the services provided,
participants in the PACE program are among the most vulnerable.  In an effort
to provide vaccinations as quickly as possible WestPACE sought to obtain the
vaccine and necessary freezer to vaccinate their members rather than wait for
San Diego County to provide them. However, obtaining a freezer was a great challenge
because at that time freezers were in high demand because of the need for
storing the vaccine. In order to satisfy the need, WestPACE could only select
from freezers that were available. The freezer obtained had a capacity which far exceeded the
needs of the center. With this excess freezer capacity, WestPace and the County
collaborated to setup an unique vaccination center with a mission to vaccinate
seniors specifically.

To meet the needs of seniors, the West family of non-profits partnered
with the local 2-1-1 organization (a non-profit that is a resource and
information hub that connects people with community, health and disaster
services). The 2-1-1 organization provided services such as a call center for the
technologically inexperienced elderly population and partnered with ride sharing services to provide
transportation to and from the vaccination site.

With these relationship in place, the vaccination clinic went from concept to
distributing vaccines in about two weeks. During its brief existence, this
clinic vaccinated thousands of seniors.

Allhough this is a  technical paper, this background describes the real impact
technology can make in peoples lives and perhaps even saving lives during one
of the most distruptive crisis in our time.

Infrastructure
--------------

The goal of the vaccine clinic is to provide accessibiliy to a senior friendly
vaccine experience. Furthermore as a non-profit and volunteer effort,
consideration must be given as to cost and manpower. Unlike well established large medical
practices, record management and Health Insurance Portability and Accountability Act (HIPAA)
compliant computer infrastructure were not well established. Even the large medical practices had
difficulty maintaining a senior-friendly environment during the early days of
the vaccine roll out when demand far exceeded capacity.

With the goal of providing a senior-friendly vaccine experience, Gary and Mary
West PACE which stood up a small senior oriented covid vaccine clinic with a desire
to mitigate the amount of paperwork to which a frail senior is
subjected. Typically several pages of information are repeatedly requested
for appointments, on consent forms, and in reminder
cards. Information ranging from basic demographics to over a dozen
health questions.

Privacy and compliance are  an important aspect of setting up a vaccine clinic
and information infrastructure for it. The key aspect ensuring
compliance to HIPAA requirements is restricting access to Protected
Health Information (PHI). For electronic systems that means all data containing PHI
should be encrypted both at rest and in transit. For paper
systems (including the printed forms mentioned below), papers containing PHI
must be not be left in the open and when unattended must be in a locked room or
container where access is restricted to authorized use. Finally, for any cloud
infrastructure the appropriate Business Associate Agreements (BAA) must be in place.

.. figure:: diagram.pdf

   Vaccination Pipeline :label:`fig:infrastructure`

Figure :ref:`fig:infrastructure` shows a high level view of the user experience and
information flow. One difficultu for the older users, especially those with
few people around them to help, is the challenge of making appointments. Because
the appointment systems were set up in a hurry, many are not well designed and confusing.
In the depicted pipeline, the persion seeking a vaccine or a caregiver would telephone the 2-1-1
call center and the operator  collects demographic and health information
during a brief interview. In addition, 2-1-1 arranges transportation to and
from the vaccine site if needed. The demographic and health information is
entered into the appointment system managed by the California Department of Public Health.
The information is downloaded from the  appointment system prior to the next day's clinic and processed
using Python for automated procedures and Jupyter for manual proceedures. (Due
to the short duration of the clinic, full automation was not deemed necessary.)
A forms packet is generated for each patient. A day's worth of packets
are then consolidated into a few PDF
files which are delivered to volunteers at the clinic, where the
volunteers print the forms. These form
packets include a consent form, county health forms, and CDC provided vaccine
cards.

When the patient arrives at the clinic, their forms are pulled, a volunteer
reviews the question with the pateint, and corrects any errors. Once the
information is validated, the patient is directed as to which forms to sign. As a
result, neither the patient nor the volunteer needs to fill the information. This
was crucial to maintain a good throughput of patients during peak times.
Generally, most patients experience less than five minute delay between arrival
at the clinic and administration of the vaccine.

While many commercial services do provide electronic form filling with electronic
signature. This system adopted paper for reasons of minimizing cost and providing a
trail for downstream audits.

Regarding compliance, some of the vaccine pipeline is handled by the third parties such as 2-1-1 or
the state. However, from the time the data is ingested from the state's
appointment system to a processing center and transmitted to the clinic,
strict HIPAA requirements are met. First, all communications from the
appointment system took place under authentication and encryption. Fortunately,
West Health has an processing center with the appropriate encryption at rest
and in transit as required by HIPAA in handling PHI. The processing
center is cloud-based but existing BAA with the cloud services were
leveraged in order to meet
HIPAA requirements. All processing took place in this
platform. Finally, the processed forms were transfered using
encryption to a server at the clinic site where an authorized operator
could securely access the forms and print them out. The paper forms
were in the custody of a volunteer until they were delivered to a back
office. Per health department regulations, the forms are then stored
for a proscribed amount of time in a locked cabinet.

Though all aspects of the pipeline faced challenges, the
pre-population of forms suprisingly posed a difficult technical
challenge due to the lack of programmatic PDF tools that properly work with
fillable forms. The remainder of the paper discusses the challenges
and provides instructions on how to use python to fill PDF forms for printing.

Programmatically Fill Forms
---------------------------

Programatically filling in PDF forms can be a quick and accurate way to
disseminate forms. Bits and pieces can be found throughout the Internet and
places like Stack Overflow. No single source provides a complete
answer. 
However, the *Medium* blog post by Vivsvaan Sharma :cite:`sharma` is a good
starting place. Another useful resource is the PDF 1.7 specification
:cite:`pdf`. Since the deployment of the vaccine clinic, the 
details of the form filling can be found at WestHealth's blog :cite:`whblog`.
The code is available on github as described below.

As a prelimiary, the following imports are used in the examples given below.

.. code:: python

    import pdfrw
    from pdfrw.objects.pdfstring import PdfString
    from pdfrw.objects.pdfstring import BasePdfName
    from pdfrw import PdfDict, PdfObject

Finding Your Way Around PDFrw and Fillable Forms
------------------------------------------------

Several examples of basic form filling code can be found on the
Internet, including the above mentioned *Medium* blog post. The
following is a typical snippet taken largely from the blog post.

.. code:: python

    pdf = pdfrw.PdfReader(file_path)
    for page in pdf.pages:
        annotations = page['/Annots']
        if annotations is None:
            continue
        
        for annotation in annotations:
            if annotation['/Subtype']=='/Widget':
                if annotation['/T']:
                    key = annotation['/T'].to_unicode()
                    print (key)

The type of ``annotation['/T']`` is ``pdfString``. While some sources use
``[1:-1]`` to extract the string from ``pdfString``, the ``.to_unicode()``
method is the proper way to extract the string. According to the PDF 1.7
specification § 12.5.6.19, all fillable forms use widget annotation.
The check for ``annotation['/SubType']`` filters the annotation
to only widget annotations.

To set the value ``value``, a ``PDFString`` needs to be created by
encoding ``value`` with the ``encode`` method. The encoded
``PDFString`` is then used to update the ``annotation`` as
shown in the following code snippet.

.. code:: python

    annotation.update(PdfDict(V=PdfString.encode(value)))

This converts `value`` into a ``PdfString`` and updates the
``annotation``, creating a value for ``annotation['/V'``].

In addition, at the top level of the ``PdfReader`` object ``pdf``, the
``NeedAppearances`` property in the interactive form dictionary,
``AcroForm`` (See § 12.7,2) needs to be set, without this, the fields are updated but
will not necessarily display. To remedy this, the following code
snippet can be used.

.. code:: python

    pdf.Root.AcroForm.update(PdfDict(
        NeedAppearances=PdfObject('true')))

Multiple Fields with Same Name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combining the code snippets provides a simple method for filling
in text fields, except if there are multiple instances of the same field. To
refer back to the clinic example, each patient's form packet comprised multiple
forms each with the ``Name`` field. Some forms even had the ``Name`` appear
twice such as in a demographic section and then in a "Print Name" field
next to a signature line.  If the code above on such a form were run,
the ``Name`` field doesn't show up. 

Whenever the multiple
fields occur with the same name the situation is more complicated. One
way to deal with this is to simply rename the fields to be different
such as ``Name-1`` and ``Name-2``, which is fine if the sole use of the
form is for automated form filling. However, if the form is also to be
used for manual filling, this would require the user to enter the
``Name`` multiple times.

When fields appear multiple times, there are some widget annotations without
the ``/T`` field but with a ``/Parent`` field. As it turns out this ``/Parent``
contains the field name ``/T`` as well as the default value ``/V``. So
for the present examples there is one ``/Parent`` and two
``/Kids``. The code can be simply modified to handle repeated fields
by inserting the following lines:

.. code:: python

    if not annotation['/T']:
        annotation=annotation['/Parent']

That can allow us to inspect and modify annotations that appear more
than once. With this modification, the result of the inspection code
yields:

.. code:: python

    pdf = pdfrw.PdfReader(file_path)
    for page in pdf.pages:
        annotations = page['/Annots']
        if annotations is None:
            continue
        
        for annotation in annotations:
            if annotation['/Subtype']=='/Widget':
                if not annotation['/T']:
                    annotation=annotation['/Parent']
                if annotation['/T']:
                    key = annotation['/T'].to_unicode()
                    print (key)

``Name`` now appears twice, once for each
instance, but they both point to the same ``/Parent``. With this
modification, the form filler will actually fill the ``/Parent`` value
twice, but this has no impact since it is overwriting the default value
with the same value.


Checkboxes
----------

In accordance to §12.7.4.2.3, the checkbox state can be set as
follows:

.. code:: python

    def checkbox(annotation, value):
        if value:
            val_str = BasePdfName('/Yes')
        else:
            val_str = BasePdfName('/Off')
        annotation.update(PdfDict(V=val_str))

This will work especially when the export value of the checkbox is
``Yes``, but doesn't need to be. The easiest solution to edit the form is to ensure that the
export value of the checkbox is ``Yes`` and the default state of the box
is unchecked. The recommendation in the specification is that it
be set to ``Yes``. In the event, the tools to make this change are not
available, the ``/V`` and ``/AS`` fields should be set to the export value
not ``Yes``.

If the form is used not only for automatic filling but manual filling,
certain checkboxes may be preferable to be checked as a default. In that case, while
the code does work, the best practice is to delete the ``/V`` as
well as the ``/AS``\ field from the dictionary. The export value can be
discovered by examining the  appearance dictionary ``/AP`` and specifically at the ``/N`` field.
Each annotation has up
to 3 appearances in its appearance dictionary: ``/N``, ``/R`` and ``/D``,
standing for *normal*, *rollover*, and *down* (§12.5.5). The latter two have to
do with appearance in interacting with the mouse. The normal appearance has to
do with how the form is printed.

According to the PDF specification for checkboxes, the appearance stream
``/AS`` should be set to the same value ``/V``. Failure to do so may
mean in some circumstances the checkboxes do not appear. It should be
noted that there isn't really strict enforcement within PDF readers, so
it is best not to tempt fate and enter a value other than the export
value for a checked value. Additionally, all these complicated
machinations with the appearance dictionary come into play when dealing
with more complex form elements.

More Complex Forms
------------------

For the purpose of the vaccine clinic application, the filling text fields
and checkboxes were all that were needed. However, in the interest of not leaving a partial
solution, other form field types were studied and solutions are given below.


Radio Buttons
~~~~~~~~~~~~~

Radio buttons are by far the most complex of the form entries types.
Each widget links to ``/Kids`` which represent the other buttons in the
radio group. But each widget in a radio group will link to the same
'kids'. Much like the 'parents' for the repeated forms fields with the
same name, each kid need only be updated each once, but it can't hurt to apply
the same update multiple times if it simplifies the code.

In a nutshell, the value ``/V`` of each widget in a radio group needs to
be set to the export value of the button selected. In each kid, the
appearance stream ``/AS`` should be set to ``/Off`` except for the kid
corresponding to the export value. In order to identify the kid with its
corresponding export value, the ``/N`` field of
the appearance dictionary ``/AP`` needs to be examined just as was
done with the checkboxes. 

The resulting code could look like the following:

.. code:: python

    def radio_button(annotation, value):
        for each in annotation['/Kids']:
            # determine the export value of each kid
            keys = each['/AP']['/N'].keys()
            keys.remove('/Off')
            export = keys[0]

            if f'/{value}' == export:
                val_str = BasePdfName(f'/{value}')
            else:
                val_str = BasePdfName(f'/Off')
            each.update(PdfDict(AS=val_str))

        annotation.update(PdfDict(
	    V=BasePdfName(f'/{value}')))

Combo Boxes and Lists
~~~~~~~~~~~~~~~~~~~~~

Both combo boxes and lists are forms of the choice form type. The combo
boxes resemble drop down menus and lists are similar to list pickers in
HTML. Functionally, they are very similar to form filling. The value
``/V`` and appearance stream ``/AS`` need to be set to their exported
values. The ``/Op`` yields a list of lists associating the exported
value with the value that appears in the widget.

To set the combo box, the value needs to be set to the export
value.

.. code:: python

    def combobox(annotation, value):
        export=None
        for each in annotation['/Opt']:
            if each[1].to_unicode()==value:
                export = each[0].to_unicode()
        if export is None:
	    err = f"Export Value: ""{value} Not Found"
            raise KeyError(err)
        pdfstr = PdfString.encode(export)
        annotation.update(PdfDict(V=pdfstr, AS=pdfstr))

Lists are structurally very similar. The list of exported values can be
found in the ``/Opt`` field. The main difference is that lists based on
their configuration can take multiple values. Multiple values can be set
with ``Pdfrw`` by setting ``\V`` and ``\AS`` to a list of ``PdfString``\ s.
The code presented here uses two separate helpers, but because of the
similarity in struction between list boxes and combo boxes, they could
be combined into one function.

.. code:: python

    def listbox(annotation, values):
        pdfstrs=[]
        for value in values:
            export=None
            for each in annotation['/Opt']:
                if each[1].to_unicode()==value:
                    export = each[0].to_unicode()
            if export is None:
	        err = f"Export Value: {value} Not Found"
                raise KeyError(err)
            pdfstrs.append(PdfString.encode(export))
        annotation.update(PdfDict(V=pdfstrs, AS=pdfstrs))

Determining Form Field Types Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the exception of the signature form (which probably should not be
filled programatically), implementation of programatic filling of all
input form field types has been presented. While PDF authoring tools
or even visual inspection can identify each forms type,
programatically determining a form field's type from the PDF document
itself would complete the package.

To address the missing ingredient, it is important to understand that
fillable forms fall into four form types, button (push button, checkboxes
and radio buttons), text, choice (combo box and list box) and signature.
They correspond to following values of the ``/FT`` form type field of
a given annotation, ``/Btn``, ``/Tx``, ``/Ch`` and ``/Sig``, respectively.
Since signature filling is not supported and push button is a widget
which can cause an action but is not fillable, those corresponding
types are omitted from consideration.

To distinguish the types of buttons and choices, the form
flags ``/Ff`` field is examined For radio buttons, the 16th bit is set. For combo
box the 18th bit is set. Please note that ``annotation['/Ff']`` returns
a ``PdfObject`` when returned and must be coerced into an ``int`` for
bit testing.

.. code:: python

    def field_type(annotation):
        ft = annotation['/FT']
        ff = annotation['/Ff']

        if ft == '/Tx':
            return 'text'
        if ft == '/Ch':
            if ff and int(ff) & 1 << 17:  # test 18th bit
                return 'combo'
            else:
                return 'list'
        if ft == '/Btn':
            if ff and int(ff) & 1 << 15:  # test 16th bit
                return 'radio'
            else:
                return 'checkbox'

For completeness, the following ``text_form`` filler helper is
included.

.. code:: python

    def text_form(annotation, value):
        pdfstr = PdfString.encode(value)
        annotation.update(PdfDict(V=pdfstr, AS=pdfstr))

This completes the building blocks to an automatic form filler.

Consolidating Multiple Filled Forms
-----------------------------------

There are two problems with consolidating multiple filled forms. The
first problem is that when two PDF files are merged matching names are
associated with each other. For instance, if John Doe were entered in
one form and Jane Doe in the second, when after combining the two forms John Doe will
override the second form's name field and John Doe would appear in both
forms. The second problem is that most simple command line or
programmatic methods of combining two or more PDF files lose form data.
One solution is to "flatten" each PDF file. This is equivalent to
printing the file to PDF. In effect, this bakes in the filled form
values and does not permit the editing the fields. Going even further,
one could render the PDFs as images if the only requirement is that the
combined files be printable. However, tools like
``ghostscript`` and ``imagemagick`` don't do a good job of preserving
form data. Other tools like PDFUnite don't solve any of these problems.

Form Field Name Collisions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The rationale for combining multiple filled PDF files arose from the
use case of the vaccine clinic. The same form was filled out for
multiple patients. But printing hundreds of individual forms was
problematic due to technological constraints (programs actually
crashed). To combine a batch of PDF forms, all form field
names are required to be different. The solution is quite
simple, in the process of filling out the form using the code above,
rename (set) the value of ``/T``.

.. code:: python

    def form_filler(in_path, data, out_path, suffix):
        pdf = pdfrw.PdfReader(in_path)
        for page in pdf.pages:
            annotations = page['/Annots']
            if annotations is None:
                continue

            for annotation in annotations:
                if annotation['/SubType'] == '/Widget':
                    key = annotation['/T'].to_unicode()
                    if key in data:
                        pdfstr = PdfString.encode(data[key])
                        new_key = key + suffix
                        annotation.update(
			    PdfDict(V=pdfstr, T=new_key))
            pdf.Root.AcroForm.update(PdfDict(
	         NeedAppearances=PdfObject('true')))
            pdfrw.PdfWriter().write(out_path, pdf)

Only a unique suffix needs to be supplied to each form. The suffix
can be as simple as a sequential number.

Combining the Files
~~~~~~~~~~~~~~~~~~~

Solutions for combining files found on the Internet for combining PDF
files using ``PDFrw``, the following recipe is typical of what ca be found.

.. code:: python

    writer = PdfWriter()
    for fname in files:
        r = PdfReader(fname)
        writer.addpages(r.pages)
    writer.write("output.pdf")

While the form data still exists in the output file, the rendering
information is lost. and won't show when displayed or printed. The
problem comes from the fact that the written PDF does not have an
interactive form dictionary (see §12.7.2 of the PDF 1.7 specification).
In particular the interactive forms dictionary contains the boolean
``NeedAppearances`` to be set in order for fields to be shown. If the
forms being combined have different interactive form dictionaries, they
will need to be merged. For the purposes here since the source
form is identical amongst the various copies, any ``AcroForm``
dictionary can be used.

After obtaining the dictionary, from ``pdf.Root.AcroForm`` (assuming the
reader is stored in ``pdf``), it is not clear how to add it to the
``PdfWriter`` object. The clue comes from a simple recipe for copying a
pdf file.

.. code:: python

    pdf = PdfReader(in_file)
    PdfWriter().write(out_file, pdf)

If one examines, these source code, the second parameter is set to the
attribute ``trailer``, so assuming ``acro_form`` contains the
interactive forms ``PdfDict`` which can be set by
``writer.trailer.Root.AcroForm = acro_form``.

Conclusion
----------

A complete functional version of this PDF form filler is open source
and can be found at WestHealth's github repository
`https://github.com/WestHealth/pdf-form-filler
<https://github.com/WestHealth/pdf-form-filler>`_ 
This process was able to produce large quantities of
pre-filled forms for seniors seeking COVID-19 vaccinations relieving one of the
bottlenecks that have plagued many other vaccine clinics.
