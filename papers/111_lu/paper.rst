
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
I ventured to get my parents vaccinated. I periodically pinged the vaccination
appointment site for a vaccine supercenter, and after week of trying, I got
through. Getting the appointment boiled down to observing the pattern of when
new appointment slots opened up. Needless to say my parents who are not
completely tech ignorant would have had extreme difficulty.

To address this the Gary and Mary West PACE (WestPACE) center established a
pop-up point of distribution (POD) for the COVID-19 vaccine :cite:`pr`
specifically for the elderly with emphasis on those who are most vulnerable.
The success in the POD was touted in the local news media :cite:`knsd`
:cite:`kpbs` and caught the attention of the state who asked WestPACE's sister
organization the Gary and Mary West Health Institute to develop a playbook for
the deploying a pop-up (POD) :cite:`pod`.

This paper gives a little more background in the effort. Next the overall
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
the county to provide them. As it turns out the freezer was a great challenge
because at this time freezers were in high demand because of their need in
storing the vaccine. In order to satisfy the need, WestPACE could only select
from freezers that were available. One that had capacity to far exceed the
needs of the center. With this excess freezer capacity, WestPace and the county
collaborated to setup an unique vaccination center with a mission to vaccinate
seniors specifically.

To meet the needs of seniors, the West family of non-profits partnered
with the local 2-1-1 organization (a non-profit that is a resource and
information hub that connects people with community, health and disaster
services.). The 2-1-1 organization provided services such as a call center for
the non-tech savvy senior and partnered with ride sharing services to provide
transportation to and rom the vaccination site.

With these relationship in place, the vaccination clinic went from concept to
distributing vaccines in about two weeks. During its brief existence, this
clinic vaccinated thousands of seniors.

Though this is a  technical paper this background describes the real impact
technology can make in peoples lives and perhaps even saving lives during one
of the most distruptive crisis in our time.

Infrastructure
--------------


The goal of the vaccine clinic is to provide accessibiliy to a senior friendly
vaccine experience. Furthermore as a non-profit and volunteer effort,
consideration as to cost and manpower. Unlike well established large medical
practices, record management and HIPAA (expand) compliant computer
infrastructure were not well established. Even the large medical practices had
difficulty maintaining a senior friendly environment during the early days of
the  vaccine roll out where demand far exceeded capacity.

With the goal of providing a senior friendly vaccine experience, Gary and Mary
West PACE which stood up a small senior oriented covid vaccine clinic desires
to mitigate the amount of paperwork a frail senior is subjected to. Quite a lot
of data is repeatedly asked for to make appointments, on consent forms and in
reminder cards.

.. figure:: diagram.pdf

   Vaccination Pipeline :label:`fig:infrastructure`

Figure :ref:`fig:infrastructure` shows at high level the user experience and
information flow. One of the great diffulties for seniors especially those with
few people around them to help is the challenge of making appointments. Because
the systems were set up in a hurry, many are not well designed and confusing.
In our pipeline, the senior or senior's caregiver would telephone the 2-1-1
call center and the operator  collects demographic and health information
during a brief interview. In addition, 2-1-1 arranges transportation to and
from the vaccine site if needed. The demographic and health information is
entered into a state maintained appointment system. The information is
downloaded the  appointment system prior to the next day's clinic and processed
using Python for automated procedures and Jupyter for manual proceedures. (Due
to the short duration of the clinic, full automation was not deemed necessary.)
A forms packet is generated for each senior and consolidated into a few PDF
files and delivered to volunteers at the clinic who print the forms. These form
packets include a consent form, county health forms and CDC provided vaccine
cards.

When the senior arrives at the clinic, their forms are pulled, a volunteer
reviews the question with the senior and corrects any errors. Once the
information is validated the senior is directed as to which forms to sign. As a
result neither the senior nor the volunteer needs to fill the information. This
was crucial for maintain a good throughput of patients during peak times.
Generally, most seniors experience less than five minute delay between arrival
at the clinic and getting the vaccine administered.

The reader may wonder why a pure electronic form system wasn't used. Many
commercial services do provide electronic form filling with electronic
signature. The reason for adopting paper is simply the cost and to provide a
trail for downstream audits.

Much of the vaccine pipeline is handled by the third parties such as 2-1-1 or
the state. However, from the time the data is ingested from the state's
appointment system to our processing center and transmitted to the clinic,
strict HIPAA requirements are met. First, all communications from the
appointment system took place under authentication and encryption. Fortunately,
West Health has an processing center with the appropriate encryption at rest
and encryption in transit as required by HIPAA in handling private health
information. All processing took place in this platform. Finally, the processed
forms were transfered to a server at the clinic site where volunteers could
securely access the forms and print them out.

Setting up most of the systems in the pipeline faced challenges. Surprisingly,
the most challenging technical difficulty was filling the forms. The remainder
of the paper discusses the challenges and provides instructions on how to use
python to fill PDF forms for printing.

While the idea of using pre-populated fillable PDF forms
is a simple one, implementation is full of challenges as many common
programmatic PDF tools do not properly work with filled forms. To meet
the challenges, PDF forms have repeated fields with same name,
checkboxes and radio buttons are used. Furthermore, to make life easier
for the staff, PDF forms for multiple patients needed to be consolidated
into a single PDF.

Programmatically Fillin Forms
-----------------------------

Programatically filling in PDF forms can be a quick and accurate way to
disseminate forms. Bits and pieces can be found throughout the Internet and
places like Stack Overflow. No single source provides a complete answer,
however, the *Medium* blog post by Vivsvaan Sharma :cite:`sharma` is a good
starting place. The blog post is long on python practices and a bit short on
PDF details. Another useful resource is the PDF 1.7 specification :cite:`pdf`
but it is well over 750 pages! Since the deployment of the vaccine clinic, the
details of the form filling can be found at our blog :cite:`whblog`, the
nitty-gritty details can be found there. The code is in the process of being
made open source and can be found at <FILLIN>.

As a prelimiary, the following imports are used in the examples given below. We
use the ``from`` directive in order to shorten the code lines so they can
easily display in this paper.

.. code:: python

    import pdfrw
    from pdfrw.objects.pdfstring import PdfString
    from pdfrw.objects.pdfstring import BasePdfName
    from pdfrw import PdfDict, PdfObject

Finding Your Way Around PDFrw and fillable forms
------------------------------------------------

If you search the internet, including the above mentioned *Medium* blog
post, you will find a snippet of code which might look like the
following:

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

The type of ``annotation['/T']`` is ``pdfString`` while some sources use
[1:-1] to extract the string from ``pdfString`` the ``.to_unicode()``
method is the proper way to extract the string. According to the PDF 1.7
specification § 12.5.6.19 all fillable forms use widget annotation,
so the check for the ``annotation['/SubType']`` filters the annotation
to only widget annotations.

To set the value, first we need to create a ``PDFString`` with
our value with the ``encode`` method then update the ``annotation`` as
shown in this code snippet.

.. code:: python

    annotation.update(PdfDict(V=PdfString.encode(value)))

This converts your ``value`` into a ``PdfString`` and updates the
``annotation`` creating a value for. ``annotation['/V'``].

As mentioned above, this won't quite do it. At the top level of your
``PdfReader`` object ``pdf`` you also need to set the
``NeedAppearances`` property in the interactive from dictionary,
``AcroForm`` (See § 12.7,2). Without this, the fields are updated but
will not necessarily display. In our example, the corresponding snippet
of code is

.. code:: python

    pdf.Root.AcroForm.update(PdfDict(
        NeedAppearances=PdfObject('true')))

Multiple Fields with Same Name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So combining the code snippets provided a simple method for filling
in text fields, except if there are multiple instances of the same field. To
refer back to the clinic example, each patient's form packet comprised multiple
forms each with the ``Name`` field. Some forms even had the ``Name`` appear
twice such as in a demographic section and then in a "Print Name" field
next to a signature line.  If we were to run the code above on such a form,
we'd find the ``Name`` field doesn't show up. 

So what happened to the ``Name`` field. Turns out whenever the multiple
fields occur with the same name the situation is more complicated. One
way to deal with this is to simply rename the fields to be different
such as ``Name-1`` and ``Name-2``, which is fine if the sole use of the
form is for automated form filling. However, if the form is also to be
used for manual filling, this would require the user to enter the
``Name`` multiple times.

When fields appear multiple times, there are some widget annotations without
the ``/T`` field but with a ``/Parent`` field. As it turns out this ``/Parent``
contains the field name ``/T`` as well as the default value ``/V``. So for our
examples there is one ``/Parent`` and two ``/Kids``. With a simple modification
to our code by inserting the lines:

.. code:: python

    if not annotation['/T']:
        annotation=annotation['/Parent']

That can allow us to inspect and modify annotations that appear more
than once. With this modification, the result of our inspection code
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

It should be noted that ``Name`` now appears twice, once for each
instance, but they both point to the same ``/Parent``. With this
modification, the form filler will actually fill the ``/Parent`` value
twice, but this has no impact since it is overwriting the default value
with the same value while keeping the code simple.


Checkboxes
----------

In accordance to §12.7.4.2.3 the you can set the checkbox state as
follows:

.. code:: python

    def checkbox(annotation, value):
        if value:
            val_str = BasePdfName('/Yes')
        else:
            val_str = BasePdfName('/Off')
        annotation.update(PdfDict(V=val_str))

This will work especially when the export value of the checkbox is
``Yes``, but doesn't need to be. The easiest solution if you designed
the form or can use Adobe Acrobat to edit the form is to ensure that the
export value of the checkbox is ``Yes`` and the default state of the box
is unchecked. The recommendation in the specification is that it
be set to ``Yes``. However, you may not have the luxury and upon closer
inspection of a form where the export value is not set to ``Yes.`` You
will see that the ``/V`` and ``/AS`` fields are set to the export value
not ``Yes``.

If you are using the form not only for automatic filling but also for manual
filling you may wish the box to be checked as a default. In that case, while
the code does work, we feel the the best solution is to delete the ``/V`` as
well as the ``/AS``\ field from the dictionary. If you do not have Acrobat and
can not find the export value, you can discover it by looking at appearance
dictionary ``/AP`` and specifically at the ``/N`` field. Each annotation has up
to 3 appearances in it's appearance dictionary ``/N``, ``/R`` and ``/D``,
standing for *normal*, *rollover*, and *down* (§12.5.5). The latter two has to
do with appearance in interacting with the mouse, the normal appearance has to
do with how the form is printed. Details on how to generalize the code to an
abritry export value can be in our blog :cite:`whblog`.

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

For the purpose of the vaccine clinic application, filling text fields
and checkboxes along with the discussion of consolidation files below
are sufficient. However, in the interest of not leaving a partial
solution. We'll take this topic further and address filling in all other
form fields.

Radio Buttons
~~~~~~~~~~~~~

Radio buttons are by far the most complex of the form entries types.
Each widget links to ``/Kids`` which represent the other buttons in the
radio group. But each widget in a radio group will link to the same
'kids'. Much like the 'parents' for the repeated forms fields with the
same name, you need only update each once, but it can't hurt to apply
the same update multiple times if it simplifies your code.

In a nutshell, the value ``/V`` of each widget in a radio group needs to
be set to the export value of the button selected. In each kid, the
appearance stream ``/AS`` should be set to ``/Off`` except for the kid
corresponding to the export value. In order to identify the kid with its
corresponding export value, we need to look again to the ``/N`` field of
the appearance dictionary ``/AP`` just as was done with the checkboxes.

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

To set the combo box, you simply need to set the value to the export
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
with Pdfrw by setting ``\V`` and ``\AS`` to a list of ``PdfString``\ s.
We code it as separate helpers, but of course, you could combine the
code into one function.

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

Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~

Now that we have shown how to fill in all the specific types of form
elements in a PDF field. (With the exception of the signature form,
which probably should not be filled programatically). Let's put this all
together. If you have access to the forms themselves, you will know what
type of form field each corresponds to each label. However, it would be
nice to be able to determine the field type and work appropriately.

Determining Form Field Types Programmatically
'''''''''''''''''''''''''''''''''''''''''''''

To address the missing ingredient, it is important to understand that
fillable forms fall into four form types, button (push button, checkboxes
and radio buttons), text, choice (combo box and list box) and signature.
They correspond to following values of the ``/FT`` form type field of
our annotation, ``/Btn``, ``/Tx``, ``/Ch`` and ``/Sig``, respectively.
We will omit the signature type as we do not support filling in
signature. Furthermore, the push button is a widget which can cause an
action but is not fillable.

To distinguish the types of buttons and choices, we can examine the form
flags ``/Ff`` field. For radio buttons, the 16th bit is set. For combo
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

For completeness, we should present a text\_form filler helper.

.. code:: python

    def text_form(annotation, value):
        pdfstr = PdfString.encode(value)
        annotation.update(PdfDict(V=pdfstr, AS=pdfstr))

So now we have all the building blocks to put an automatic form filler
together. The finished form filler can be found in our Github repository
at github.com/westhealth.

Consolidating Multiple Filled Forms
-----------------------------------

There are two problems with consolidating multiple filled forms. The
first problem is that when two PDF files are merged matching names are
associated with each other. For instance, if I had John Doe entered in
one form and Jane Doe in the second, when I combine them John Doe will
override the second form's name field and John Doe would appear in both
forms. The second problem is that most simple command line or
programmatic methods of combining two or more PDF files lose form data.
One solution is to "flatten" the each PDF file. This is equivalent to
printing the file to PDF. In effect, this bakes in the filled form
values and does not permit the editing the fields. Going even further,
one could render the PDFs as images if the only requirement is that the
combined files be printable. However, tools like
``ghostscript`` and ``imagemagick`` don't do a good job of preserving
form data. Other tools like PDFUnite don't solve any of these problems.

Form Field Name Collisions
~~~~~~~~~~~~~~~~~~~~~~~~~~

In our use case of the vaccine clinic, we have the same form being
filled out for multiple patients. So to combine a batch of these
requires all form field names to be different. The solution is quite
simple, in the process of filling out the form using the code above, we
can also rename (set) the value of ``/T``.

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

So all you have to do is supply a unique suffix to each form. In our
case, we simply number the batch so the suffix is just a sequential
number.

Combining the files
~~~~~~~~~~~~~~~~~~~

If you search the internet for combine PDF files using pdfrw, you'll get
a recipe like the following.

.. code:: python

    writer = PdfWriter()
    for fname in files:
        r = PdfReader(fname)
        writer.addpages(r.pages)
    writer.write("output.pdf")

While you don't lose the form data per se, you lose rendering
information and hence the combined PDF fails to show the fields. The
problem comes from the fact that the written PDF does not have an
interactive form dictionary (see §12.7.2 of the PDF 1.7 specification).
In particular the interactive forms dictionary contains the boolean
``NeedAppearances`` to be set in order for fields to be shown. If the
forms being combined have different interactive form dictionaries, they
will need to be merged. For our purposes since the source
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
interactive forms ``PdfDict`` you can set it by
``writer.trailer.Root.AcroForm = acro_form``.

Conclusion
----------

A complete functional version of this PDF form filler can be found in our
github repository. This process was able to produce large quantities of
pre-filled forms for seniors seeking COVID-19 vaccinations relieving one of the
bottlenecks that have plagued many other vaccine clinics.
