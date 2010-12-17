:author: Kristopher Overholt
:email: koverholt@gmail.com
:institution: The University of Texas at Austin

---------------------------------------------------------
Numerical Pyromaniacs: The Use of Python in Fire Research
---------------------------------------------------------

.. figure:: cssci_art.jpg

.. class:: abstract

 Python along with various numerical and scientific libraries have been used to create tools that enable fire protection engineers to perform various calculations and tasks including educational instruction, experimental work, and data visualization. This is only the beginning of the utilization of Python in this field, and I hope to connect with the greater scientific Python community to foster a plethora of new ideas and innovation in the field of fire protection research and engineering.

Introduction
------------

As much as fire protection engineers like to protect people from fire, we also really like to burn stuff. When we are not burning things or blowing stuff up, we deal with data analysis, numerical simulations, and a global engineering community.

The use of Python in fire protection engineering and fire research has many useful applications and allows for an abundance of possibilities for the fire science community. Recently, Python has been used to create data analysis tools for fire experiments and simulation data, to assist users with fire modeling and fire dynamics calculations, to perform flame height tracking for warehouse fires and bench-scale tests using GUI programs, and to present flame spread results using a novel visual method that involves superimposed video plotting. These tools have emerged in the form of web applications, GUIs, and data reduction scripts which interact with existing computational fluid dynamics (CFD) tools such as the freely available and open-source fire modeling program, Fire Dynamics Simulator (FDS) (http://fire.nist.gov/fds), which is maintained by the National Institute of Standards and Technology (NIST) in the U.S.

Python (and its associated scientific & numerical libraries) is the perfect tool for the development of these tools, which  advance the capabilities of the fire research community, due to its free and open source nature and widespread, active community. Thus, it is important to identify efforts and projects that allow engineers and scientists to easily dive into learning Python programming and to utilize it in an environment that is familiar and efficient for practical engineering use.

Tools and packages such as the Enthought Python Distribution, Sage, and Python (x,y) certainly ease the installation of Python and its associated scientific and numerical packages. Other tools such as Spyder and Sage allow engineers and students to work in a familiar environment that is similar to commercial engineering programs while empowering them with the advantages of using a true programming language (along with powerful libraries) that is based on an open-source foundation.

The following sections describe applications in which Python has been used in the fire research and fire protection engineering field. First, some common applications of Python are detailed with respect to the use of Python for data analysis and visualization. Then, the process of the creation of engineering web tools for fire protection engineering is given. This is followed by an overview of how Python is being used along with CFD fire models. Finally, the future plans for utilizing Python in fire research are given.

Interactive Data Analysis for Engineers
---------------------------------------
I started using Python a few years ago after becoming frustrated with the shortcomings, inflexibility, and poor cross-platform performance of popular data analysis software and engineering tools. As an engineering student, when I switched to Python, I wondered if I was making a mistake by working with any tool other than the de facto standard for engineering calculations and data analysis. Now, years later, I know that it was the right choice and will continue exploring the ways that Python can empower my field.

Through the advent of tools such as Spyder, Python(x,y), PythonToolkit, which recreate an interactive data analysis and rapid-prototyping environment that engineers are familiar with, the use of Python as an engineering tool continues to grow.

In addition previously mentioned reasons for using Python, I enjoy using Python in my workflow because of its enormous amount of flexibility. If I quickly create a data reduction tool or calculator and want to develop it further, I don't have to rewrite my existing code. Rather, I can incrementally wrap additional code around and share it either as a GUI application or as a web application (an example is presented later).

Another great application of Python in fire research is for the verification and validation of fire models. As with other CFD tools, the developers of fire models have recently become interested in the use of verification and validation to assess the performance and accuracy of the models regarding their correct physical representation. This process typically involves comparing the results from the model to analytical results or experimental test data and usually involves a copious amount of data and plots.

We can use Python to automate our workflow and generate plots from updated data runs, and we are considering using Python to generate regression statistics between newer versions of the fire modeling software so that we can better quantify the numerical variations that are related to changes in the CFD code.

From data fitting to generating beautiful plots using matplotlib, I feel that Python is quickly emerging as a viable tool for engineers and scientists to interact with data in a meaningful and programmatic manner that is familiar to them. An example plot of experimental data and a fit from Python that was used in my master's thesis is shown as:

.. figure:: mlr_trimmed.png

I believe that the development of interactive tools such as Spyder and Sage will continue to grow (the projects are gaining a large amount of momentum), and students will continue to explore how Python can enhance their research experience due to the abilities of rapid prototyping and data exploration, the opportunity to easily create GUI or web tools, and the ability to effortlessly share their data and programs without wondering if the recipient has the required proprietary software.

In an era where a budget cut can result in numerous dropped licenses for proprietary data analysis software (which can cause the proprietary scripts and data sets to be unusable), Python is in a great position to become more ubiquitous in fire protection engineering, and the engineering field in general. I also believe that a tool such as Python will also help speed along the transition to the era of open publishing and open data due to its open nature and strong community.

Fresh Data Visualization Methods
--------------------------------
Every once in a while, someone makes an impression on you that lasts for a lifetime. This story shares one of those times, and this is one that can change the way you present information in a very meaningful way.

A few years ago, I was attending a fire conference, and someone that was working on a project regarding the structural response of buildings on fire showed a video in their presentation. At a typical fire engineering conference, we get to see cool fire and explosion videos as well as plots in the presentations. Sometimes the plots are interesting; more often they are default plots from common spreadsheet software with an ugly legend and boring colors – with no story to tell.

But not this time. The presenter showed a video with real-time seismic data plots superimposed over a video showing the actual burning building. “Amazing!”, I thought, and it stuck with me. It was certainly a useful way to convey information in parallel and make the data tell a story. Because people can gather more meaning from videos rather than plots, why not tell the qualitative AND quantitative story at the same time?

After the conference, I worked on writing a script that would import videos, adjust frame rates, plot on the imported figures, and so on. I used it to show video plots of warehouse commodity fire tests with actual and predicted flame heights vs. time:

.. figure:: warehouse_vid.png

I also used the script to show video plots of the predicted flame heights for a small-scale test in an way that just about anyone can relate to, whether you're a fire-crazed scientist or not:

.. figure:: bench_vid.png

Real-time video plots are a great visual method for teaching, communication, and telling a story with your data. Surprisingly enough, I haven’t found any existing programs or tools that do this. Python was the perfect tool for this, and, as described in the previous section, it wouldn't take much more effort on my part to create a GUI for this tool and release it.

Web Tools and Engineering Calculators
-------------------------------------
Data analysis and visualization is common to many different fields, and I wanted to move into bigger and better uses of Python in fire protection engineering. As I was learning Python a few years ago and using it in my research, I created a tool that would help me generate a mesh for a numerical fire simulation that involved some simple, yet tedious calculations and Poisson-friendly number requirements. After doing these monotonous calculations by hand one too many times, I wrote a Python script to help me generate the correct numbers and parameters to use in my input file, and I thought about how I could share this tool with others. After digging around some simple Python CGI examples, I created an online mesh size calculator. Amazing! I had never created a web application before, and it was easy, AND fun! 

The calculator interface is shown as:

.. figure:: mesh_calc.png

Today, on my website, the tool gets used about 500 times a month by engineers and scientists around the world. Often, when I am attending conferences, a stranger will gaze at my name tag for a few seconds, then greet me and thank me for the mesh size calculator tool. The tool is available at: http://www.koverholt.com/fds-mesh-size-calc.

The results of the calculator are shown as:

.. figure:: mesh_calc2.png

After this wonderful experience, I cannot stop the ideas and possibilities from flowing. I continuously dream up new tools and calculators that could easily be created with the use of Python, such as a suite of fire engineering and fire dynamics tools that can be used online. For example, there is a program called FPEtool (fire protection engineering tool), which contains a set of fire dynamics calculations and was heavily used in the 1980s and 1990s. It is still available for free from NIST - as a DOS executable. Because of this, the use of the excellent tools and fire dynamics calculators in FPEtool are no longer used in the field. I think it would be great to revive FPEtool as a web-based, open-source, and community supported project using Python. In conclusion, Python offers our field the ability to easily and quickly create web tools, from simple calculators to complex web applications, and this results in a more efficient workflow for engineers, a method for third-party developers to contribute to the fire modeling community, and promotion of the effective use of fire dynamics and tools for life safety designs.

Creating 3D Geometry for Fire Models
------------------------------------
Regarding the increasing amount of interaction between Python and fire models, some third-party developers in the fire modeling community (including myself) have recently released a tool to model 3D geometry and generate a text-based input file for the fire modeling software, FDS. The tool is called BlenderFDS, and is an extension for Blender that was created using Python. Before the release of BlenderFDS, users of FDS had to create geometry for a case using either a text editor or an expensive commercial GUI. Now, using BlenderFDS, users can create complex buildings and irregular geometry (e.g., cylinders, angled roofs) and automatically have it broken up into the rectilinear format that FDS requires. The interface for the BlenderFDS extension in Blender is shown as:

.. figure:: testcase_obj2obst

BlenderFDS allows for the quick creation of complex geometry in a visual manner, and it can even be used to model the complex geometry of an entire building:

.. figure:: fds2.jpg

We hope to continue adding functionality to BlenderFDS to result in a comprehensive GUI for creating input files for fire models, and we (the developers) have appreciated the ease of use and the implementation process of Python interaction with Blender for this project. More information about the BlenderFDS project can be found at http://www.blenderfds.org. We also continue to explore additional solutions in Blender and other popular CFD postprocessing tools, which will be discussed in the next section.

Visualizing Smoke and Fire for CFD simulations
-----------------------------------------------
With the availability of numerous CFD-related tools such as Paraview, Mayavi, and Blender, we have been exploring the use of these tools for the visualization of realistic and scientifically-meaningful fire and smoke from the results of CFD fire simulations. An example of realistic fire in the upcoming release of Blender 2.5 is shown (from Andrew Price, blenderguru.com) as:

.. figure:: campfire.jpg

Not only would such a visualization tool allow for graphical improvements in the output, but it would also allow for a standard format for visualization and analysis, which exists in many other fields that utilize CFD simulations. Finally, such a tool would also allow for community involvement and support for the visualization software.

Future Plans for Python in Fire Research
----------------------------------------

The use of Python in fire protection engineering is still in its early stages; future applications in the fire research field include: a web interface for fire dynamics and engineering calculation tools, tools to analyze and visualize output from CFD programs such as FDS, and the design and implementation of a standardized open format and database for experimental fire test data.

Interactive data analysis tools that are based on Python, such as Spyder and Sage, will allow Python to further penetrate into the engineering field as a flexible, free, and powerful tool that is backed a supportive, active community. To push Python further into engineering use, more emphasis should be placed on the development interactive analysis and GUI tools to create a viable alternative to commercial engineering and scientific software.

Python can be further utilized in tools such as Blender (for geometry creation), Spyder (for interactive data analysis and scripting), or Mayavi (for visualization), which allows for the possibility of many new innovations in fire research while allowing the field to advance upon the ideological values of free and open-source software. Finally, Python can be further incorporated into the world of CFD and high performance computing environments.

In conclusion, the use of Python in science and engineering is of utmost importance to us because fire protection engineering and fire research involve public safety and strive to produce safer buildings and materials to protect people and property around the world from the dangers of fire. I have had a more than pleasurable experience working with Python and the scientific Python community, and I hope to interact with the community even more to explore possibilities and create even more solutions that can advance our field.
