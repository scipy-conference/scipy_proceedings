:author: Josh Walawender
:email: joshw@naoj.org
:institution: Subaru Telescope, National Astronomical Observatory of Japan

---------------------------------------------
Automated Image Quality Monitoring with IQMon
---------------------------------------------

.. class:: abstract

Automated telescopes are capable of generating images more quickly than they can be inspected by a human, but detailed information on the performance of the telescope is valuable for monitoring and tuning of their operation.  The IQMon (Image Quality Monitor) package[#]_ was developed to provide basic image quality metrics of automated telescopes in near real time. 

.. [#] Source code available at https://github.com/joshwalawender/IQMon

.. class:: keywords

   astronomy, automated telescopes, image quality

Introduction
------------

Using existing tools such as astropy [astropy]_, astrometry.net, source extractor, SCAMP, and SWARP, IQMon analyzes images and provides the user a quick way to determine whether the telescope is performing at the required level.

IQMon can provide provide a determination of whether the telescope is focused (from the typical FWHM of stars in the image), whether it is pointing accurately (obtained from a comparison of the target coordinates with the astrometrically solved coordinates), whether the tracking or guiding is adequate (from the typical ellipticity of stars in the image), and whether the night is photometric (obtained from the typical photometric zero point of stars in the image).  For wide field systems which detect many stars in each image, these metrics can be spatially resolved allowing for more detailed analysis such as differentiating between tracking error, focus error, and optical aberration or determining if the dome is partially obscuring the telescope aperture.

Because the system is designed to do quick evaluations of image quality, the primary concept is an object representing a **single** image.  It does not do any image stacking or other processing which would be applied to more than one image at a time nor is it built around other organizational concepts such as targets or visits.  It is not intended to supplant a full data reduction and analysis package.  The output of IQMon, however, can be stored in a MongoDB[#]_ making it potentially useful for collecting information on observing concepts such as targets, nights, or visits which span multiple images.  It might also be useful as a preprocessing step for a more complex data pipeline.

.. [#] http://www.mongodb.org

To date, IQMon has been deployed on three disparate systems: a 735mm focal length wide field imager with a monochrome CCD camera which undersamples the PSF, an 0.5 meter f/8 telescope with a monochrome CCD camera with well sampled PSF, and an 85mm focal length camera lens and DSLR camera (with Bayer color array) designed for very wide field photometry.  IQMon has provided valuable diagnostic information about system performance in all cases.


Structure and Example Use
-------------------------

IQMon operates by using ``Telescope`` and ``Image`` classes.  The ``Telescope`` object contains basic information about the telescope which took the data.  When a ``Telescope`` object is instantiated, a configuration file is read which  contains information on the telescope and controls various user configurable parameters and preferences for IQMon.  The configuration file is a YAML document and is read using the ``yaml`` module.

An ``Image`` object is instantiated with a path to a file with one of the supported image formats and with a reference to a ``Telescope`` object.  The image analysis process is simply a series of calls to methods on the ``Image`` object.

The IQMon philosophy is to never operate on the raw file itself, but instead to create a "working file" (using the ``read_image`` method) and store it in a temporary directory.  If the raw image file is a FITS file, then ``read_image``  simply copies the raw file to the temporary directory and records this file name and path in the ``working_file`` property.  If the file is a raw image file from a DSLR (e.g. ``.CR2`` or ``.dng`` format), then ``read_image`` will call ``dcraw`` using the subprocess32 module to convert the file to ``.ppm``.  The file is then converted to FITS format using either ``pamtofits`` or ``pnmtofits`` tools from the ``netpbm`` package.  To date IQMon has only been tested with FITS and ``.CR2`` files, but should in principle work with numerous DSLR raw format images.

In the following sections, I will describe a simple example of evaluating image quality for a single image.  A more complex example which is updated in concert with IQMon can be found in the ``measure_image.py`` script at the git repository for the VYSOS project[#]_.

.. [#] https://github.com/joshwalawender/VYSOStools

After importing IQMon, the first step would be to instantiate the ``Telescope`` object which takes a configuration file as its input.  The next step is to instantiate an ``Image`` object with the path to the image file and the ``Telescope`` object representing the telescope which took that image.

.. code-block:: python

    tel = IQMon.Telescope('~/MyTelescope.yaml')
    im = IQMon.Image('~/MyImage.fits', tel)

IQMon writes a log which is intended to provide useful information to the user (not just the developer) and shows the progress of the analysis.  We can either pass in a ``logger`` object from Python's logging module, or ask IQMon to create one:

.. code-block:: python

    im.make_logger(verbose=False) # create a new logger object
    print('Logging to file {}'.format(im.logfile))
    im.logger.info('This is a log entry')

The first step for any image analysis is likely to be to call the ``read_image`` method.  After calling ``read_image``, the FITS header is read and various ``Image`` object properties are populated by calling the ``read_header`` method.

.. code-block:: python

    im.read_image()   # Generate working file copy of the raw image
    im.read_header()  # Read the fits header

Once the image has been read in and a working file created, IQMon uses various third party tools to perform image analysis.  The following sections describe some of the analysis steps which are available.


PSF Size Measurements with Source Extractor
```````````````````````````````````````````

Source Extractor (SExtractor; [Bertin1996]_[Bertin2010_SExtractor]_) is called using the ``run_SExtractor`` method which invokes the command using the subprocess32 module.  Customization parameters can be passed to Source Extractor using the telescope configuration file.

The output file of SExtractor is read in and stored as an astropy table object.  Stars with SExtractor generated flags are removed from the table and the table is stored as a property of the image object.

Determining the PSF size from the SExtractor results is done with the ``determine_FWHM`` method.  The full width at half maximum (FWHM) and ellipticity values for the image are a weighted average of the FWHM and ellipticity values for the individual stars.

..
    with weights :math:`w_i = (F_i / \sigma_{F_i})^2`.  Where :math:`w_i` is the weight assigned to that star's FWHM value, :math:`F_i` is the flux the star, and :math:`\sigma_{F_i}` is the uncertainty in that flux.  These values correspond to the FLUX_AUTO and FLUXERR_AUTO values reported by SExtractor.  These weights are then used as input to the ``np.average`` weight keyword to determine the weighted average FWHM for the image.  

These steps not only provide the typical FWHM (which can indicate if the image is in focus), they can also be used to guess at whether the image is "blank" (i.e. very few stars are visible either because of cloud cover or other system failure).  For example:

.. code-block:: python

    im.run_SExtractor()
    # Consider the image to be blank if <10 stars detected
    if im.n_stars_SExtracted < 10:
        im.logger.warning('Only {} stars found. Image may be blank.'\
                             .format(im.n_stars_SExtracted))
    else:
        im.determine_FWHM()


Pointing Determination and Pointing Error
`````````````````````````````````````````

IQMon also contains a ``solve_astrometry`` method to invoke the ``solve-field`` command which is part of the astrometry.net [Lang2010]_ software.  The call to ``solve-field`` is only intended to determine basic pointing and orientation and so deactivates the SIP polynomial fit of distortion in the image.

Once a world coordinate system (WCS) is present in the image header, then the ``determine_pointing_error`` method can be called which compares the right ascension (RA) and declination (DEC) values read from the RA and DEC keywords in the header (which are presumed to be the telescope's intended pointing) to the RA and DEC values calculated for the center pixel using the WCS.  The separation between the two coordinates is reported as the pointing error.

.. code-block:: python

    # If WCS is not present, analyze the image with astrometry.net,
    if not im.image_WCS:
        im.solve_astrometry()
        im.read_header()
    # Solve for the pointing error by comparing the telescope
    # pointing coordinates from the header with the WCS solution.
    im.determine_pointing_error()

Astrometric Distortion Correction
`````````````````````````````````

In order to make an accurate comparison of the photometry of stars detected in the image and stars present in a chosen stellar catalog, many optical systems require distortion coefficients to be fitted as part of the astrometric solution.  IQMon uses the SCAMP [Bertin2006]_ [Bertin2010_SCAMP]_ software to fit distortions.

SCAMP is invoked with the ``run_SCAMP`` method.  Once a SCAMP solution has been determined, the image can be remapped to new pixels without distortions using the SWARP [Bertin2010_SWARP]_ tool with the ``run_SWARP`` method.

.. code-block:: python

    # If the image has a WCS and a SExtractor catalog, run SCAMP to
    # determine a WCS with distortions.
    if im.image_WCS and im.SExtractor_results:
        im.run_SCAMP()
        if im.SCAMP_successful:
            im.run_SWarp()   # Remap the pixels to a rectilinear grid
            im.read_header() # Update the header

Estimating the Photometric Zero Point
`````````````````````````````````````

With a full astrometric solution, SExtractor photometry, and a catalog of stellar magnitude values, we can estimate the zero point for the image and use that as an indicator of clouds or other aperture obscurations.

The ``get_catalog`` method can be used to download a catalog of stars from VizieR using the astroquery module.  Alternatively, support for a local copy of the UCAC4 catalog is available using the ``get_local_UCAC4`` method.  Once a catalog is obtained, the ``run_SExtractor`` method is invoked again, this time with the ``assoc`` keyword set to ``True``.

.. code-block:: python

    im.get_catalog()  # Retrieve catalog defined in config file
    im.run_SExtractor(assoc=True)
    im.determine_FWHM()
    im.measure_zero_point()

In the above example code, ``determine_FWHM`` is invoked again in order to use the new SExtractor catalog for the calculation.

The ``measure_zero_point`` method determines the zero point by taking the weighted average of the difference between the measured instrumental magnitude from SExtractor and the catalog magnitude in the same filter.  

..
    The weights for each measurement are assumed to be :math:`w_i = (\frac{ln(10)}{2.512} \, \frac{F_i}{\sigma_{F_i}})^2`.  Using these weights, the zero point for the image is calculated using the ``np.average`` method.

Flags
`````

For the four primary measurements (FWHM, ellipticity, pointing error, and zero point), the configuration file may contain a threshold value.  If the measured value exceeds the threshold (or is below the threshold in the case of zero point), then the image is "flagged" as an indication that there may be a potential problem with the data.  The flags property of an ``Image`` object stores a dictionary with the flag name and a boolean value as the dictionary elements.

This can be useful when summarizing results.  The Tornado web page provided with IQMon, for example, lists images and will color code a field red if that field is flagged.  In this way, a user can easily see when and where problems might have occurred.

JPEGs and Plots
```````````````

In addition to generating single values for FWHM, ellipticity, and zero point to represent the image, IQMon can also generate more detailed plots with additional information.

A plot with PSF quality information can be generated when ``determine_FWHM`` is called by setting the ``plot=True`` keyword.  This generates a .png file using matplotlib which shows detailed information about the point spread function (FWHM and ellipticity metrics) including histograms of individual values, a spatial map of FWHM and ellipticity over the image, and plots showing the ellipticity vs. radius within the image (which can be used to show whether off axis aberrations influence the ellipticity measure) and the correlation between the measured PSF position angle and the position angle of the star within the image (which can be used to differentiate between tracking error and off axis aberrations).

A plot with additional information on the zero point can be generated when calling ``measure_zero_point`` by setting the ``plot`` keyword to ``True``.  This generates a .png file using matplotlib which shows plots of instrumental magnitude vs. catalog magnitude, a histogram of zero point values, a plot of magnitude residuals vs. catalog magnitude, and a a spatial map of zero point over the image.

JPEG versions of the image can be generated using the ``make_JPEG`` method.  The jpeg can be binned or cropped using the ``binning`` or ``crop`` keyword arguments and various overlays can be generated showing the pointing error and detected and catalog stars.


Storing Results and Mongo Database Integration
``````````````````````````````````````````````

Results of the IQMon measurements for each image can be stored for later use.  Methods exist to write them to an ``astropy.Table`` (the ``add_summary_entry`` method) and to a YAML document (the ``add_yaml_entry`` method), but the preferred storage solution is to use a mongo database.

The address, port number, database name, and collection name to use with pyMongo to add the results to an existing mongo database are set by the Telescope configuration file.  The ``add_mongo_entry`` method adds a dictionary of values with the results of the IQMon analysis.

Tornado Web Application
```````````````````````

IQMon comes with a tornado web application which, while it can be run stand alone, is intended to be used as a template for adding IQMon results to a more customized web page.  The web application (``web_server.py``) contains two ``tornado`` web handlers: ``ListOfNights`` and ``ListOfImages``.  The first generates a page which lists UT dates and if there are image results associated with a date, then it provides a link to a page with the list of image results for that date.  The second handler produces the page which lists the images for a particular UT date (or target name) and provides a table formatted list of the IQMon measurement results for each image with flagged values color coded red, along with links to jpegs and plots generated for that image.

..
    Summary
    ```````

..
    IQMon provides a simple toolkit for evaluating image quality.  While it could be used in other applications, it was intended to be a tool for evaluating the performance of robotic telescopes.


References
----------
.. [astropy] Astropy Collaboration, Robitaille, T.~P., Tollerud, E.~J., et al.
             *Astropy: A community Python package for astronomy* 2013, A&A, 558,
             A33

.. [Bertin1996] Bertin, E., & Arnouts, S. *SExtractor: Software for source
                extraction*, 1996, A&AS, 117, 393

.. [Bertin2006] Bertin, E. *Automatic Astrometric and Photometric Calibration
                with SCAMP*, 2006, Astronomical Data Analysis Software and
                Systems XV, 351, 112

.. [Bertin2010_SCAMP] Bertin, E. *SCAMP: Automatic Astrometric and Photometric
                      Calibration*, 2010, Astrophysics Source Code Library,
                      1010.063

.. [Bertin2010_SExtractor] Bertin, E., & Arnouts, S. *SExtractor: Source
                           Extractor*, 2010, Astrophysics Source Code Library,
                           1010.064

.. [Bertin2010_SWARP] Bertin, E. *SWarp: Resampling and Co-adding FITS Images
                      Together* 2010, Astrophysics Source Code Library, 1010.068

.. [Lang2010] Lang, D., Hogg, D. W., Mierle, K., Blanton, M., & Roweis, S.,
              *Astrometry.net: Blind astrometric calibration of arbitrary
              astronomical images* 2010, AJ 137, 1782–1800
