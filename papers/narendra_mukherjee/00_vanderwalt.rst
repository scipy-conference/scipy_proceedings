:author: Narendra Mukherjee
:email: narendra@brandeis.edu
:institution: Graduate Program in Neuroscience, Brandeis University, Waltham, MA, USA
:corresponding:

:author: Joseph Wachutka
:email: wachutka@brandeis.edu
:institution: Graduate Program in Neuroscience, Brandeis University, Waltham, MA, USA

:author: Donald B Katz
:email: dbkatz@brandeis.edu
:institution: Department of Psychology, Brandeis University, Waltham, MA, USA
:institution: Volen National Center for Complex Systems, Brandeis University, Waltham, MA, USA

:bibliography: mybib

:video: https://github.com/narendramukherjee/blech_clust

--------------------------------------------------------------------------------------------------------------------
Python meets systems neuroscience: affordable, scalable and open-source electrophysiology in awake, behaving rodents
--------------------------------------------------------------------------------------------------------------------

.. class:: abstract

Electrophysiology, the recording of neurons in the brains of awake, behaving animals, is currently undergoing paradigm shifts as a basic experimental technique in systems neuroscience. There is a push towards moving to open-source technologies that can be adjusted to specific experiments, can be shared with ease, are affordable and can record from large numbers of electrodes simultaneously. Here we describe our approach to setting up such a system using the scientific Python stack and Linux – we use a Raspberry Pi to control experimental paradigms and build a completely open-source, HDF5-based analysis (spike sorting) toolkit in Python that can be easily parallelized on HPCs or standalone computers and scales with increasing electrode counts and longer recordings. Our setup costs about $5000, an order of magnitude less than comparable commercially available electrophysiology systems that cost in the range of $100k.   

.. class:: keywords

   electrophysiology, Python, open-source, HDF5   

Introduction
------------

Recording neural populations in awake, behaving animals (in-vivo extracellular electrophysiology) is key to understanding the brain and its coordination of complex behaviors. It involves voltage recordings from bundles of microwire electrodes (10-20 microns in diameter) surgically implanted into the brain regions of interest. In-vivo electrophysiology has been widely used to understand brain function and behavior in a wide range of animal species from invertebrates (locusts and grasshoppers – REF) to fishes (REF), birds (REF), rodents (REF) and primates (REF). Electrophysiological studies in awake, behaving animals provide an unprecedented view of the complex and highly variable brain dynamics that underlie accurate behavioral responses. In-vivo electrophysiology is, therefore, unique in a neuroscientist’s toolkit in the degree of resolution it provides at both the spatial and temporal (sub-millisecond) levels – it has yielded insights into brain structure and function ranging from the cellular (some LTP examples) to the systems (some taste/hippocampus refs) levels.

Despite its seminal importance in the experimental neuroscience stack, electrophysiology hardware and software has classically been plagued by proprietary and closed-source technologies. These closed-source designs are not amenable to being modified to suit specific experimental circumstances, and go against the philosophy of open and reproducible science by making it hard for other investigators to replicate experimental results. Above all, proprietary electrophysiology hardware and software is prohibitively expensive, and poses a high ‘barrier to entry’ for neuroscientists starting to set up their laboratories or working under tight budgetary constraints in developing nations. As a consequence of the use of closed-source technologies, in-vivo electrophysiology hardware and software has been virtually unchanged for the last 20 years despite electronics and computing taking giant strides forward in that time.

With reproducible and affordable science in mind, some electrophysiology laboratories have recently started to push towards completely open source electrophysiology hardware and software (open ephys REF). There is an added impetus in modern electrophysiology towards increasing ‘channel counts’ - recording from hundreds, or even thousands, of electrodes implanted in several different brain regions to understand the inter-regional coordination that underlies brain function and animal behavior. New electrophysiological hardware and software, apart from being open-source, affordable and reproducible, must therefore ‘scale’ with growing experiment and data sizes.

In this paper, we propose that the scientific Python stack is apt for all these challenges being faced by modern electrophysiology. We describe a completely open-source, Python-based hardware and software setup that we use to study the role of gustatory (taste) cortex in taste-related learning and behavior in rats. We use a Raspberry Pi based system to coordinate the various behavioral paradigms of our experiments – this includes the delivery of precise amounts of taste solutions to the animals :cite:`katz2002taste` and the optogenetic perturbation of the firing of neurons in the taste cortex by pulsing laser sources at 20-30Hz :cite:`li2016sensory` :cite:`pastrana2011optogenetics`. At the electrophysiological level, we use open-source electrophysiology chips from Intan Technologies (Intan REF) and develop an HDF5 and Python-based software setup for spike sorting (REF) and analysis. We describe the computations involved at every step of our spike sorting toolchain and highlight software principles that make such an analysis setup 1.) scale with increased channel counts and longer recordings and 2.) easily parallelized on computing environments. We demonstrate the use of this system to record and analyze electrophysiological data from 64 electrodes simultaneously in the taste cortex of rats and mice. Finally, we compare and contrast our approach to the traditional electrophysiology and spike sorting toolchain and point out future directions of improvement keeping the modern electrophysiology experiment in mind.

Animal care, handling and surgeries
-----------------------------------

We used adult, female Long-Evans rats (325-300g) and adult mice (JOE FILL THIS UP) in our experiments. They were surgically implanted with bundles of microwire electrodes bilaterally in the gustatory (taste) cortex and intra-oral cannulae (IOCs) behind the cheek for delivering taste solutions. All animal care and experiments complied with the Brandeis University Institutional Animal Care and Use Committee guidelines. For more details on experimental protocols, see :cite:`sadacca2016behavioral`

Raspberry Pi based behavior control system
------------------------------------------

We use a Raspberry Pi running Ubuntu MATE (https://ubuntu-mate.org/blog/ubuntu-mate-xenial-point-2-raspberry-pi/) to weave together the various behavioral paradigms of our experiments. This includes delivering precise amounts of taste solutions to the animals by controlling pressurized solenoid valves, measuring the animals’ licking responses with an analog-to-digital converter (ADC) circuit and pulsing laser sources for optogenetic perturbation at 20-30Hz. Most of these steps involve controlling the digital I/O pins (DIO) of the Pi – the Rpi.GPIO package provides convenient functions:

.. code-block:: python
    
    import RPi.GPIO as GPIO
    # The BOARD mode allows referring to the GPIO pins 
    # by their number on the board
    GPIO.setmode(GPIO.BOARD)
    # Set port 1 as an output
    GPIO.setup(1, GPIO.OUT)
    # Send outputs to port 1
    GPIO.output(1, 1)
    GPIO.output(1, 0)
    
Electrode bundles and microdrives
---------------------------------

We build *opto-trode* bundles with 32 microwires (DIAMETER) and a 200 \mu fiber for optogenetics per bundle and 3D print microdrives from Shapeways. Our custom built drives cost about $60, compared to over $800 for a comparable proprietary design (http://neuronexus.com/products/neural-probes/optogenetics/optoelectrode). These designs are freely available for use and modification at (Electrode link on Katz lab website) and (Shapeways link).

Electrophysiology hardware
--------------------------

We use open-source electrophysiology headstages from Intan Technologies for neural recordings (Intan RHD link). These headstages plug into the electrode bundles implanted in the animal’s brain and contain 32-128 amplifiers and ADCs. The Intan data acquisition system offers an open-source C++ based graphical interface that can record upto 512 electrodes (4 headstages) simultaneously at sampling rates of upto 30kHz. This recording system is relatively robust to AC noise (as the electrode signals are digitized right on the headstage itself) – we additionally encase the animal’s behavior and recording chamber in a Faraday cage constructed with standard aluminum insect netting.

We have used this system to record from 64 electrodes (2 bundles of 32 wires each placed bilaterally in the gustatory cortex, numbered as ports A and B) at 30kHz for upto 2 hours at a time in our rats. We split the recordings into individual files for each electrode channel and digital input – this enables us to organize the recorded files by electrode number and keep each file restricted to a reasonable size (about 200-300MB).

Scientific Python stack for data analysis – spike sorting
---------------------------------------------------------

The recent push in electrophysiological experiments towards increased channel counts and longer recordings poses significant data handling and analysis challenges. Each of the implanted electrodes needs to be sampled at frequencies in the range of 20-30kHz to be able to detect action potentials (the signature binary voltage waveforms, about 1ms in duration, that neurons produce when active – also called ‘spikes’, hence the name ‘spike sorting’). In our experiments, we sample 64 electrodes at 30kHz for upto 2 hours, generating datasets that total 10-30GB in size. Datasets of such sizes cannot be loaded into memory and processed in serial – there is evidently a need to convert the data to a format that allows access to specific parts of the data and can support a parallel computing framework.

The Hierarchical Data Format (HDF5) is ideal for dealing with such big numerical datasets. We use the Pytables package (http://www.pytables.org/) to build, structure and modify HDF5 files at every point in our spike sorting and analysis toolchain. Pytables allows data to be stored and extracted from HDF5 files in the convenient form of numpy arrays.  We decided to use every individual electrode as a storage and computation split – the voltage recording from every electrode is then stored as a separate array in the HDF5 file with its analysis assigned to a separate process.

We adopted a semi-supervised approach to spike sorting starting with a (parallelized) set of automated filtering and clustering steps that are finally tuned by the experimenter with their expert knowledge about action potential shapes actually observed in the brain. Our setup therefore involves 3 distinct steps (all the code is available on Github at https://github.com/narendramukherjee/blech_clust):

1. Pre-processing (**blech_clust.py**) – Constructs a HDF5 file with the raw binary data recorded by the Intan system, acquires the clustering parameters from the user and creates a shell file that runs the actual processing step in parallel.
2. Processing (**blech_process.py**) – Runs filtering and clustering steps on the voltage data from every electrode and plots out the results.
3. Post-processing (**blech_post_process.py**) – Removes raw recordings from the HDF5 file and compresses it, and then allows the user to sieve out real spikes from the putative spikes plotted in step 2.

Pre-processing
--------------

The pre-processing step starts by building a HDF5 file for the electrophysiology dataset with separate nodes for raw electrodes, digital inputs and outputs. This structuring of different aspects of the data into separate nodes is a recurrent feature of our toolchain – we hope that such an organization of data within a single file will make collaboration and data-sharing easier. The Pytables library provides a convenient set of functions for this purpose:

.. code-block:: python

    # modified from blech_clust.py
    import tables
    # Create hdf5 file, and make group for raw data
    hf5 = tables.open_file(hdf5_name[-1]+'.h5', 'w',
              title = hdf5_name[-1])
    hf5.create_group('/', 'raw')
    hf5.close()
    
We now set up Pytables **extendable arrays** (EArrays) to read the electrode and digital input data saved by the Intan system. Extendable arrays are akin to standard Python lists in the sense that their size can be ‘extended’ as data is appended to them – unlike lists, however, they are a homogeneous data class and cannot store different types together. The Intan system saves all the data as integers in binary files and therefore, EArrays of type int (defined by IntAtom in Pytables) are perfect for this purpose. These EArrays can be constructed and filled as follows:

.. code-block:: python

    # Modified from create_hdf_arrays() in read_file.py
    # Open an existing HDF5 file with read and write permissions - r+
    hf5 = tables.open_file(file_name, 'r+')
    # 2 headstages each with 32 electrodes in our experiments
    n_electrodes = len(ports)*32
    # All the data is stored as integers
    atom = tables.IntAtom()
    # Create arrays for neural electrodes
    for i in range(n_electrodes):
    	el = hf5.create_earray('/raw', 'electrode%i' % i, atom, (0,))
    hf5.close()
    
    # Modified from read_files() in read_file.py
    # Open HDF5 file with read and write permissions - r+
    hf5 = tables.open_file(hdf5_name, 'r+')
    # Fill data from electrode 1 on port A
    # Electrode data are stored in binary files
    # as 16 bit signed integers
    data = np.fromfile('amp-A-001.dat', 
                       dtype = np.dtype('int16')) 
    hf5.flush()
    hf5.close()
    
All through the spike sorting process, we use the easygui package (http://easygui.readthedocs.io/en/master/) to integrate user inputs through a simple graphical interface. Finally, we use GNU parallel :cite:`Tange2011a` to run filtering and clustering on every electrode in the dataset in a separate process. GNU parallel is a great parallelization tool on .nix systems, and allows us to 1.) assign a minimum amount of RAM to every process as well as to 2.) resume failed processes by reading from a log file.

Processing
----------

The voltage data from the electrodes are stored as signed integers in the HDF5 file in the pre-processing step – they need to be converted into actual voltage values (in \muV) as floats. The datasheet of the Intan RHD2000 system (http://intantech.com/files/Intan_RHD2000_series_datasheet.pdf) gives the transformation as:

.. math::
   
    voltage (\mu V) = 0.195 * voltage (int)

Spikes are high frequency events that typically last for 1-1.5 ms – we therefore remove low frequency transients by bandpass filtering the data in 300-3000 Hz using a 2-pole Butterworth filter as follows:

.. code-block:: python

    # Modified from get_filtered_electrode()
    # in clustering.py
    from scipy.signal import butter
    from scipy.signal import filtfilt 
    m, n = butter(2, [300.0/(sampling_rate/2.0),
                  3000.0/(sampling_rate/2.0)], 
                  btype = 'bandpass') 
    filt_el = filtfilt(m, n, el)

At this point, typical spike sorting toolchains involve imposing an amplitude threshold on the voltage data to detect spikes – depending on the position of the electrode in relation to neurons in the brain, action potentials appear as transiently large positive or negative deflections from the mean voltage detected on the electrode. The wide swath of action potentials from extracellularly recorded cortical neurons are appear as negative voltage deflections from the average – we therefore define a threshold based on the electrode’s median voltage (:cite:`quiroga2004unsupervised`) and choose the segments of the recording that go below it:

.. code-block:: python

    # Modified from extract_waveforms() in clustering.py
    m = np.mean(filt_el)
    th = 5.0*np.median(np.abs(filt_el)/0.6745)
    pos = np.where(filt_el <= m–th)[0]

We treat each of these segments as a potential spike – we locate the minimum of each segment and slice out 1.5ms (0.5ms before the minimum, 1ms after = 45 samples at 30kHz) of data around it. Even at the relatively high sampling rates that we use in our experiments, it is possible that these segments are significantly ‘jittered’ in time and their shapes do not line up exactly. In addition, we pick up a large number of segments that have multiple troughs (or minima) and are definitely not spikes. To deal with these scenarios, we ‘dejitter’ the set of potential spikes by interpolating their shapes (using *scipy.interpolate.interp1d*), up-sampling them 10-fold using the interpolation, and finally picking just the segments that can be lined up by their unique minimum. These 450-dimensional ‘putative spikes’ will now be clustered by fitting a Gaussian Mixture Model (GMM) :cite:`lewicki1998review`. The user eventually picks the best solution with their expert knowledge in the manual part of our semi-automated spike sorting toolchain.

Each of the putative spike waveforms picked above consists of 450 samples after interpolation – there can be more than a million such waveforms in a 2 hour recording from each electrode. We, therefore, reduce the dimensionality of the dataset by picking the first 3 components produced through principal component analysis (PCA) :cite:`bro2014principal` using the scikit-learn package :cite:`scikit-learn`. These principal components, however, are known to depend mostly on the amplitude-induced variance in shapes of recorded action potential waveforms – to address this possibility, we scale each waveform by its energy (modified from :cite:`Fee1996175`), defined as follows, before performing the PCA:

.. math::
    	
    Energy = \frac{1}{n} \sqrt{\sum_{i=1}^{450} X_i^{2}}\ where\ X_i\ =\ i^{th}\ component\ of\ the\ waveform

Finally, we feed in the energy and maximal amplitude of each waveform as features into the GMM in addition to the first 3 principal components. Using scikit-learn’s GMM API, we fit GMMs with cluster numbers varying from 2 to a user-specified maximum number (usually 7 or 8). Each of these models is fit to the data several times (usually 10) and the best fit is chosen according to the Bayesian Information Criterion (BIC) :cite:`bhat2010derivation`. 

The clustering results need to be plotted for the user to be able to pick action potentials from the ‘noise’ in the post-processing step. The most important in these sets of plots are the actual waveforms of the spikes clustered together by the GMM and the distribution of their inter-spike-intervals (ISIs) (more details in the post-processing step). Plotting the waveforms of the putative spikes in every cluster produced by the GMM together, however, is the most memory-expensive step of our toolchain. For a 2 hour recording with 64 electrodes, the plotting step with matplotlib :cite:`Hunter:2007` can consume upto 6GB of memory although the PNG files that are saved to disk are only of the order of 100KB. High memory consumption during plotting also limits the possibility of applying this spike sorting framework to recordings that are several hours long – as a potential substitute, we have preliminarily set up a live plotting toolchain using Bokeh (http://bokeh.pydata.org/en/latest/docs/dev_guide.html) that can be used during the post-processing step. We are currently trying to work out a more memory-efficient plotting framework, and any suggestions to that end are welcome.

Post-processing
---------------

Once the parallelized processing step outlined above is over, we start the post-processing step by first deleting the raw electrode recordings (under the ‘raw’ node) and compressing the HDF5 file using ptrepack (http://www.pytables.org/usersguide/utilities.html) as follows:

.. code-block:: python

    # Modified from blech_post_process.py 
    hf5.remove_node('/raw', recursive = True)
    # Use ptrepack with compression level = 9 and
    # compression library = blosc
    os.system("ptrepack --chunkshape=auto --propindexes 
              --complevel=9 --complib=blosc " + hdf5_name
              + " " + hdf5_name[:-3] + "_repacked.h5")
    
The logic of the post-processing step revolves around allowing the user to look at the GMM solutions for the putative spikes from every electrode, pick the solution that best splits the noise and spike clusters, and choose the cluster numbers that corresponds to spikes. The GMM clustering step, being unsupervised in nature, can sometimes put the spikes from two (or more) separate neurons (with very similar energy-scaled shapes, but different amplitudes) in the same cluster or split the spikes from a single neuron across several clusters. In addition, the actual action potential waveform observed on a electrode depends on the timing of the activity of the neurons in its vicinity – co-active neurons near an electrode can additively produce spike waveforms that have smaller amplitude and are noisier (called ‘multi’ units) (Figure :ref:`fig3`) than single, isolated neurons (called ‘single’ units). Therefore, we set up utilities to merge and split clusters in the post-processing step – users can choose to merge clusters when the spikes from a single neuron have been distributed across clusters or split (using a GMM clustering using the same features as in the processing step) a single cluster if it contains spikes from separate neurons. 

HDF5, once again, provides a convenient format to store the single and multi units that the user picks from the GMM results. We make a ‘sorted_units’ node in the file to which units are added in the order that they are picked by the user. In addition, we make a ‘unit_descriptor’ table that contains metadata about the units that are picked – these metadata are essential in all downstream analyses of the activity of the neurons in the dataset. To setup such a table through Pytables, we first need to create a class describing the datatypes that the columns of the table will hold and then use this class as the description while creating the table.

.. code-block:: python

    # Modified from blech_post_process.py
    # Define a unit_descriptor class to be used 
    # to add things (anything!) about the sorted
    # units to a pytables table
    class unit_descriptor(tables.IsDescription):
    	electrode_number = tables.Int32Col()
    	single_unit = tables.Int32Col()
    	regular_spiking = tables.Int32Col()
    	fast_spiking = tables.Int32Col()
    
    # Make a table describing the sorted units. 
    # If unit_descriptor already exists, just open it up
    try:
    	table = hf5.create_table('/', 'unit_descriptor', 
    	                    description = unit_descriptor)
    except:
    	table = hf5.root.unit_descriptor
    
Cortical neurons (including gustatory cortical neurons that we record from in our experiments) fall into two major categories – 1.) excitatory pyramidal cells that define cortical layers and have long range connections across brain regions, and 2.) inhibitory interneurons that have short range connections. In extracellular electrophysiological records, pyramidal cells produce relatively large and slow action potentials at rates ranging from 5-20 Hz (spikes/s) (Figure :ref:`fig1`). Interneurons, on the other hand, have much higher spiking rates (upto 50-70 Hz) and much faster (and hence, narrower) action potentials (Figure :ref:`fig2`). Therefore, in the unit_descriptor table, we save the type of cortical neuron that the unit corresponds to in addition to the electrode number it was located on and whether its a single unit (Figure :ref:`fig3`). In keeping with classical electrophysiological terminology, we refer to putative pyramidal neuron units as ‘regular spiking units (RSU)’ and interneuron units as ‘fast spiking units (FS)’ :cite:`mccormick1985comparative` :cite:`hengen2013firing`. In addition, anatomically, pyramidal cells are much larger and more abundant than interneurons in cortical regions (REF) – expectedly, in a typical gustatory cortex recording, 60-70% of the units we isolate are RSUs. This classification of units is in no way restrictive – new descriptions can simply be added to the unit_descriptor class to account for recordings in a sub-cortical region that contains a different electrophysiological unit.

Apart from the shape of the spikes (look at Figures :ref:`fig1`, :ref:`fig2`, :ref:`fig3`, :ref:`fig4` to compare spikes and typical noise) in a cluster, the distribution of their inter-spike-intervals (ISIs) (plotted in the processing step) is another important factor in differentiating single units from multi units or noise. Due to electrochemical constraints, after every action potential, neurons enter a ‘refractory period’ - most neurons cannot produce another spike for about 2ms. We, therefore, advise a relatively conservative ISI threshold while classifying single units – in our recordings, we designate a cluster as a single unit only if <0.01% (<1 in 10000) spikes fall within 2ms of another spike.

Finally, we consider the possibility that since the processing of the voltage data from each electrode happens independently in a parallelized manner, we might pick up action potentials from the same neuron on different electrodes (if they are positioned close to each other). We, therefore, calculate ‘similarity’ between every pair of units in the dataset – this is the percentage of spikes in a unit that are within 1ms of spikes in a different unit. This metric should ideally be very close to 0 for two distinct neurons that are spiking independently – in our datasets, we consider units that have similarity greater than 20% as the same neuron and discard one of them from our downstream analysis. To speed up this analysis, especially for datasets that have 20-40 neurons each with <10000 spikes, we use Numba’s just-in-time compilation (JIT) feature (http://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html):

.. code-block:: python

    # Modified from blech_units_distance.py
    from numba import jit
    @jit(nogil = True)
    def unit_distance(this_unit_times, other_unit_times):
    	this_unit_counter = 0
    	other_unit_counter = 0
    	for i in range(len(this_unit_times)):
    		for j in range(len(other_unit_times)):
    			if np.abs(this_unit_times[i]
    			          - other_unit_times[j]) <= 1.0:
    				this_unit_counter += 1
    				other_unit_counter += 1
    	return this_unit_counter, other_unit_counter
    	
Conclusions
-----------

In-vivo extracellular electrophysiology in awake, behaving animals provides a unique glimpse into the activity of populations of neurons in the brain that underlie the animals’ behavioral responses to complex stimuli. Recording, detecting, analyzing and isolating action potentials of single neurons in a brain region in an awake animal poses a variety of technical challenges, both at the hardware and software levels. Rodent and primate electrophysiologists have classically used proprietary hardware and software solutions in their experiments – these closed-source technologies are expensive, not suited to specific experimental contexts and hard to adapt to sharing and collaboration. With the push towards open, collaborative and reproducible science, modern electrophysiological needs are urging open-source experimental setups that are affordable and scale with the growing sizes of electrophysiogical experiments. In this paper, we have outlined a Raspberry Pi and scientific Python-based solution to these technical challenges and successfully used it to perform electrophysiological recordings and optogenetics in the gustatory (taste) cortex of awake mice and rats. Our setup can scale as data sizes grow with increasingly longer recordings and larger number of electrodes, and costs ~$5000 (compared to at least $100k for a comparable proprietary setup).

Our approach uses the HDF5 data format at its heart that allows us to arrange all the data (and their associated metadata) under specific nodes in the same file. This approach has several advantages over the traditional electrophysiological data structure of several files in a proprietary format: firstly, HDF5 is a widely used cross-platform data format that has convenient APIs in all major programming languages. Secondly, having all the data from an experimental session in the same file (that can be easily compressed – we use ptrepack in the post-processing step) makes data sharing and collaboration easier. Thirdly, HDF5 files allow quick access to desired parts of the data during analysis – as a consequence, larger than memory workflows can easily be supported without worrying about the I/O overhead involved. Lastly, in our setup, we splice the storage and processing of the data by individual electrodes – this allows us to run the processing step in parallel on several electrodes together bringing down processing time significantly.

Our semi-automated approach to spike sorting is faster and more principled than the standard approach of picking units by 1.) placing an arbitrary, user-defined amplitude threshold on spike waveforms during the recordings and 2.) manually drawing polygons around spikes from a unit in principal component (PC) space. We automate both these steps of the traditional spike sorting toolchain by using an amplitude threshold that depends on the median voltage recorded on an electrode and clustering putative spikes with a Gaussian Mixture Model (GMM). The user’s knowledge only feeds in the last step of our workflow to label the clusters picked out by the GMM as noise, single unit or multi unit based on the shapes of the spike waveforms and their ISI distributions. As the number of electrodes in an electrophysiological recording start running into the hundreds and thousands, there is a need to automate this last manual step as well – this can be achieved by fitting supervised classifiers to the units (and their types) picked out manually in a few training datasets. As the waveforms of spikes can depend upon the brain region being recorded from, such an approach would likely have to applied to every brain region separately.

During the pre-processing step, we restrict our setup to only pick ‘negative’ spikes – where the voltage deflection goes ‘below’ a certain threshold. While most extracellular spikes will appear as negative voltage deflections (as they are being mostly recorded from outside the axons of neurons), sometimes an electrode, depending on the brain region, ends up being close enough to the cell body of a neuron to record positive spikes. The pre-processing step, in such a scenario, will need trivial modifications to include positive deflections ‘above’ a threshold as spikes as well.

Due to the use of the HDF5 format and the ease of supporting larger-than-memory workflows, our toolchain will scale with modern electrophysiology that is pushing towards longer recordings and increased electrode counts. However, as explained previously, plotting all the spike waveforms in a cluster together during the processing step using matplotlib is a major memory bottleneck in our workflow. We are working on figuring out a more efficient workaround, and have devised a live plotting setup with Bokeh (that plots 50 waveforms at a time) that can be used during post processing instead. In addition, recordings running for several hours (or days) have to account for the change in spike waveforms induced by ‘electrode drift’ - the electrode moves around in the fluid medium of the brain with time. The live plotting module is potentially useful in such longer recordings as well – it can be used to look at spikes recorded in small windows of time (30 minutes say) to see if their shapes change with time.

We are currently attempting to fold up our Python based electrophysiology analysis setup into the format of a Python package that can be used by electrophysiologists (using the Intan recording system) to analyze their data with ease on a shared computing resource or on personal workstations. We think that using the scientific Python will make previously hidden ‘under the hood’ spike sorting principles clearer to the average electrophysiologist, and will make implementing downstream analyses on these data easier. 

.. figure:: Unit19.png

   A regular spiking unit (RSU) - 45 samples (1.5ms) on the x axis. Note the 2 inflection points as the spikes go back to baseline from their minimum :label:`fig1`

.. figure:: Unit11.png

   A fast spiking unit (FS) - 45 samples (1.5ms) on the x axis. Compare to :ref:`fig1` and note that this unit has narrower/faster  spikes and has higher firing rate (more spikes) :label:`fig2`

.. figure:: Unit13.png

   A multi unit - 45 samples (1.5ms) on the x axis. Compare to :ref:`fig1` and :ref:`fig2` and note that these spikes have smaller amplitudes and are noisier :label:`fig3`

.. figure:: Cluster4_waveforms.png

   A noise cluster - 45 samples (1.5ms) on the x axis. :label:`fig4`

   	

References
----------


