:author: Dallin Skouson
:email: dallinskouson@byu.edu
:institution: NSF Center for Space, High-Performance, and Resilient Computing (SHREC)
:institution: Department of Electrical and Computer Engineering, Brigham Young University

:author: Andrew Keller
:email: andrewmkeller@byu.edu
:institution: NSF Center for Space, High-Performance, and Resilient Computing (SHREC)
:institution: Department of Electrical and Computer Engineering, Brigham Young University
:orcid: 0000-0002-6285-5288

:author: Michael Wirthlin
:email: wirthlin@byu.edu
:institution: NSF Center for Space, High-Performance, and Resilient Computing (SHREC)
:institution: Department of Electrical and Computer Engineering, Brigham Young University
:bibliography: mybib
:orcid: 0000-0003-0328-6713

--------------------------------------------------------------
Performing Netlist Analysis and Transformations Using SpyDrNet
--------------------------------------------------------------

.. class:: abstract

   Hardware circuits can contain a large number of discrete components and connections. These connections are defined by
   a data strauture called a "netlist". Important information can be gained by analyzing the structure of the circuit 
   netlist and relationships between components. Many specific circuit manipulations require component reorganization in
   hierarchy and various circuit transformations.

   SpyDrNet is an open-source netlist analysis and transformation tool that performs many of these functions written in 
   Python. It provides a framework for netlist representation, querying, and modification. This tool is actively used to
   enhance circuit reliability in radiation environments through partial circuit replication.

   Circuit representations come in a variety of formats, each having unique requirements. SpyDrNet aims to provide the
   flexibility needed to meet the demands of a wide variety of structural netlist representations.

.. class:: keywords

   Hardware Design, Netlists, SpyDrNet, EDIF

Introduction
------------

Hardware circuits can contain a large number of discrete components and connections. These components work together 
through their connections to implement a hardware design. Hardware circuits can be found on printed wiring boards or 
in very large scale integration (VLSI) as implemented in integrated circuits. Discrete components can be analog or 
digital in nature and each component and connection can be associated with specific attributes. All of this information
can be stored inside a graphlike data structure called a "netlist" which details each component, its attributes, and 
connections.

Netlists come in many different formats and organizational structures, but common constructs abound (within EDIF, 
structural Verliog, and structural VHDL, etc.) :cite:`edif_based,verilog_netlist`. Most netlist formats have a notion of
primitive or basic circuit components that form a basis from which any design can be created. If the contents of a 
circuit component is unknown, it is treated as a blackbox. Primitive or basic components and blackboxes are viewed as leaf 
cells, modules, or definitions, which can then be instanced individually into larger non-leaf definitions. These 
definitions contains wires, nets, buses, or cables that together connect ports or pins on instances or on the definition
itself. Instancing definitions within definitions provides hierarchy up to the top hierarchical instance and definitions
can be further organized into libraries to keep things neat and tidy.

SpyDrNet provides a common framework for representing, querying, and modifying netlists from which application specific
analysis and transformation functions can be built. The data structure used to represent netlists is designed to provide
quick pointer access to neighboring elements and it is designed to be extensible so that format specific constructs can 
be stored along with the netlist for preservation when the netlist is exported. These constructs allow for the 
representation of a wide variety of netlist formats. SpyDrNet differs from most related tools
in that its focus is on structural netlists as opposed to the synthesis or simulation of hardware description languages.

SpyDrNet is currently implemented in pure Python and provides a Python interface so that it can easily integrate with
other Python packages such as NetworkX :cite:`networkx` and PyEDA :cite:`pyeda`. These library packages have been used
in tandem with SpyDrNet to prototype new analysis techniques for better understanding the connectivity and 
relationships between circuit components as part of reliability research. The Python platform also makes this tool 
readily available to anyone in the community that may be interested in using it. 

This paper presents the SpyDrNet framework, some of its use cases, and highlights its use in the development of 
advanced reliability enhancement techniques. This tool originates from a long line of reliability research focused on
improving the reliability of computer circuits implemented on static random access memory based (SRAM-based) field programmable gate arrays (FPGAs)
:cite:`johnson_dwc,pratt_2008,Johnson:2010`. The predecessor to this tool is BYU EDIF Tools :cite:`BYUediftools`. 
Development efforts moved towards SpyDrNet to bring the previous tools functionality to Python and open up its use to a
larger number of formats and unique applications.

.. figure:: spydrnetflow3.png
   :scale: 30%
   :align: center
   :figclass: w

   The path of a design using SpyDrNet. :label:`exteriorfig`

Related Work
------------

SpyDrNet is not the first tool of its kind. The predecessor to SpyDrNet, BYU EDIF Tools :cite:`BYUediftools`, is a Java 
based tool released in 2008 intended primarily for use with netlists targeting FPGAs produced by Xilinx. Xilinx itself 
offers a robust tool command language (TCL) scripting environment for querying and modifying a netlist among other 
specialized implementation tasks; a custom CAD tool framework has taken advantange of this environment :cite:`tincr`. A 
tool similar to SpyDrNet built for hardware description languages (HDLs) is LiveHD :cite:`livehd`. LiveHD is an 
infrastructure focused on the synthesis and simulation of HDLs. It looks more at the whole design cycle (from synthesis,
to simulation, to place and route, and tapeout) with rapid turnaround for small changes, but it may also feasibly be 
used to work with structural netlists. A tool more specific to Xilinx FPGA implementation is RapidWright 
:cite:`rapid_wright`. It also contains a netlist representation, and is taylored towards low-level physical 
implementation.


SpyDrNet Tool Flow
------------------

Electronic designs may be converted a number of times before they are ready to be built, packaged, or programmed into their target device. For example, these designs may be created in a hardware description language, synthesized into a netlist, then placed, routed, and packged into a target file which will be used to fabricate the device. A CAD tool can begin to modify the functionality of the final design at various of these stages. The earlier stages in the design flow are slightly less static. Constructs may be optimized out of the design, and the actual hardware implementation of a construct may be unknown. Later in the design process these things are more stable, but the design is also less easy to work with (binary files, complex device specific information, etc). By working at the netlist level, SpyDrNet is able to avoid many of the pitfalls of both aspects of the design process. 

Figure :ref:`exteriorfig` represents how a design can be prepared and processed prior to and after using SpyDrNet. Many designs start as a hand written hardware description language and are then converted into a netlist using a synthesizer. Netlists are then passed through additional tools to create a design file to be physically implemented

Internally the SpyDrNet tool is composed of a flow that begins with a parser, accepting any of the supported languages. The parser creates an in memory data structure of the design stored in the intermediate representation. After this the tool can perform any of its analysis or modification passes on the design. Once the design is in a state where the user is satisfied a supported export function called a composer is used to pass the design back out. Figure :ref:`flowfig` represents the internal flow within SpyDrNet.

.. figure:: flow.png
   :align: center
   :figclass: htbp

   Universal representation capabilities of the intermediate representation, Note that Verilog and VHDL refer to the structural subset of these languages :label:`flowfig`

The Intermediate Representation
-------------------------------

The intermediate representation is a generic structural netlist representation employed by SpyDrNet.  Structural 
netlists refer to a class of netlists that represent circuit components but not necessarily their behaviour. These 
netlists are useful because when modifying netlists for reliability we are less concerned with the general purpose of 
the circuit and more concerned with how that circuit is implemented. SpyDrNet’s internal intermediate representation is 
an in-memory construct. Users can manipulate the structure while in memory and write out a supported format using one of
the export modules or *composers* that is included with SpyDrNet. More advanced users with special requrements could 
also create their own composer to support the format that they desire. The API is complete enough to support full parser
and composer support. Users need not learn excess information about the internals of the netlist to create an effective
composer or parser.

SpyDrNet aims to be programmer friendly. The datastructure was built with a focus on simplifying access to adjacent points in the netlist. In some cases where simple accessors could be added at additional memory cost, the accessors were added. One example of this is the bidirectional references implemented throughout the netlist. This ideology resulted in a slightly longer running time in some cases (and shorter in others), but speed was taken into account as these decisions were made. If a feature significantly increased the run time of the tests, it was examined and optimized.


Constructs Employed
*******************


A short description of some of the datastructure components is provide here to help the reader more easily visualize how optimization trade offs were selected. This background will also assist as some of the core functionality of SpyDrNet is later discussed. The constructs behind a structural Netlist are Libraries, Definitions, Instances, Ports, and Cables. Figure :ref:`irfig` shows the connectivity between these components. 

.. figure:: IR.png
   :align: center
   :figclass: htbp

   Highlights the connectivity between components in the intermediate representation :label:`irfig`


**Element**
+++++++++++

This is the base class for all components of a netlist. Components are further divided into first class elements and regular elements. First class elements have a name field as well as a properties field.

**Definition**
++++++++++++++

These objects are sometimes called cells or modules in other representations. They hold all of the information about what their instances contain.

**Instance**
++++++++++++

This element is a place holder to be replaced with the subelements of the corresponding definition upon build. It is contained in a different definition to its own. In the case of the top level instance it is the place holder that will be replaced by the entire netlist when it is implemented

**Port**
++++++++

The Port element can be thought of as containing the information on how a Definition connects the outside world to the elements (instances and cables) it contains.

**Cable**
+++++++++

Cables are bundles of wires that connect components within a definition. They connect ports to their destination pins.

**Pin**
+++++++

These objects represent points of connection between instances or ports and wires. Pins can be divided into inner and outer pin categories. The need for these distinctions lies in the fact that definitions may have more than one instance of itself. Thus components connected on the inside of a definition need to connect to pins related to the definition will connect to inner pins on the definition. Each of these inner pins will correspond to one or more outer pins on instances of the corresponding definition. In this way instances can be connected togehter while still allowing components within a definition to connect to the ports of that definition.

**Wire**
++++++++

Wires are grouped inside cables and are elements that help hold connection information between single pins on instances within a definition and within it’s ports.

.. figure:: ExampleCircuit.png
   :align: center
   :figclass: htbp

   Structure of the Intermediate Representation :label:`egfig`

Extensible Support for Multiple Netlist Formats
***********************************************

In addition to holding a generic netlist data structure, the universal netlist representation can hold information specific to individual formats. This is done through the inclusion of metadata dictionaries in many of the SpyDrNet objects. 

Parsers can take advantage of the flexibility of the metadata dictionary to carry extra information that source formats present through the tool. This includes information such as comments, parameters, and properties.

In addition, the metadata dictionary can be used to contain any desired user data. Because the tool is implemented in python, any data type can be used for the key value in these dictionaries, however we only guarantee future support of string objects.

Callback Framework
------------------

Some potential use cases for SpyDrNet could involve making incremental changes to the netlist, and following each of them up with an analysis of the netlist to determine what more needs to changed. Alternatively users may wish to be warned of violations of design rules such as maintaining unique names. These checks could be performed over the whole netlist datastructure on user demand which would add complexity for the end user. To fill this gap a callback framework was implemented.

These callbacks allow users to create plugins that can keep track of the current state of the netlist as changes are made. Currently, a namespace manager is included with SpyDrNet. The callback framework is able to watch changes to the netlist, including addition and removal of elements, as well as changes in namming and structure of the netlist.

Listeners may register to hear these changes as they happen. Each listener is called in the order in which it was registered and may update itself as it sees the netlist change. Plugins that implement listeners can be created and added through the API defined register functions. In general listener functions are expected to receive the same parameters as the function on which they listen.


Modularity by design
********************

In order to support expansion to a wide variety of netlists, our intermediate representation was designed to reflect a generic netlist data structure. Care was taken to ensure that additional user defined constructs could be easily included in the netlist.

Additionally, to maintain modularity, the intermediate representation can be built entirely using the existing API calls. These calls also allow the netlist to be written out or composed after modification. The existing parsers and composers use the API to achieve their functions.

Because of the generic nature of the netlist representation and the ability to build it using only the API additional netlist parsers and composers can be built separately and still take full advantage of the existing modification passes available in SpyDrNet. To build a parser or composer requires no more advanced knowledge than an end user may have from using the API to design a custom analysis or modification pass on the netlist.

Other functionality has been added on top of the core of SpyDrNet, including plugin support and the ability to modifiy the netlist at a higher level. These utility functions are used by applications. This layered approach aims to aid in code reusability and reliability allowing lower level functionality to be tested before the higher level functionality is added on.


Analysis and Transformation Capabilities
----------------------------------------

SpyDrNet was created with FPGA reliability in mind. One current application of SpyDrNet focuses on implementing duplication with compare (DWC) and triple modular redundancy (TMR) to circuit designs. Some of the design considerations that go into effect while choosing a tool to implement these reliability modifications, include avoiding optimizations, and algorithmic modification capability. It is desirable to have a flexible framework. Additionally behavioral modifications are not generally needed because the structural implementation is simple enough to be easily implemented directly.

SpyDrNet grew to fill these needs. Modifications made with SpyDrNet are less likely to be optimized away. Additionally SpyDrNet allows users to create custom algorithms that will modify components of the netlist. Modifications are done at the structural level which is simple for our reliability algorithms of interest.

Utility Functions
-----------------

SpyDrNet has several high level features currently included. All of these features have an impact on the overall netlist structure but several are most useful when included in other applications. This section will highlight some of the simpler high level features that are currently implemented in SpyDrNet. 

Basic Functionality
*******************
Functionality is provided through the API to allow for creation and modification of elements in the netlist datastructures. Sufficient functionality is provided to create a netlist from the ground up, and read all available information from a created netlist. Netlist objects are completely mutable and allow for on demand modification. This provides a flexible framework upon which users can build and edit netlists data structures. The basic functionality includes functionality to create new children elements, modify the properties of elements, delete elements, and change the relationships of elements. All references bidirectional and otherwise are maintained behind the scenes to ensure the user can easily complete modification passes on the netlist while maintaining a valid representation.

The mutability of the objects in SpyDrNet is of special mention. Many frameworks require that the object's name be set on creation, and disallow any changes to that name. SpyDrNet, on the other hand, allows name changes as well as any other changes to the connections, and properties of the objects. The callback framework, as discussed in another section, provides hooks that allow checks for violations of user defined rules if desired.

Examples of some of the basic functionality are highlighted in the following code segment. Relationships, such as the reference member of the instances and the children of these references are members of the spydrnet objects. Additional key data can be accessed as members of the classes. Other format specific data can be accessed through dictionary lookups. Since the name is also key data but, is not required it can be looked up through either access method as noted in one of the single line comment.

.. code-block:: python
   
   import spydrnet as sdn

   netlist = sdn.load_example_netlist_by_name(
      'fourBitCounter')
   
   top_instance = netlist.top_instance
  
   def recurse(instance, depth):
      '''print something like this:
      top
         child1
             child1.child
         child2
             child2.child'''
      s = depth * "\t"
      
      #instance.name could also be instance["NAME"]
      print(
         s, instance.name,
         "(", instance.reference.name, ")")
      for c in instance.reference.children:  
         recurse(c, depth + 1)
   
   recurse(top_instance, 0)

Hierarchy
*********

Hierarchy is by default a component of many netlist formats. One of the main advantages to including hierarchy in a design is the ability to abstract away some of the finer details on a level based system, while still including all of the information needed to build the design. The design’s hierarchical information is maintained in SpyDrNet by having definitions instanced within other definitions.

SpyDrNet allows the user to work with the structure of a netlist directly, having only one of each instance per hierarchical level, but it also allows the user view the netlist instances in a hierarchical context through the use of hierarchical references as outined below. Some other tools only provide the hierarchical representation of the design.

There are drawbacks and advantages to each view on the netlist, but the inclusion of a hierarcical view helps allow users to make the fewest possible unneeded changes to the design. Additionally there are several advantages to maintaining hierarchy, smaller file sizes are possible in some cases, as sub components do not need to be replicated. Simulators may have an easier time predicting how the design will act once implemented :cite:`build_hierarchy`. Further research could be done to analyze the impact of hierarchy on later compilation steps.

Flattening
**********

SpyDrNet has the ability to flatten hierarchical designs. One method to remove hierarchy from a design is to move all of the sub components to the top level of the netlist repeatedly until each sub component at the top level is a terminal instance, where no more structural information is included below that instance’s level.

Flattening was added to SpyDrNet because there are some algorithms which can be applied more simply on a flat design. Algorithms in which a flat design may be simpler to work with are graph analysis, and other algorithms where the connections between low level components are of interest.

Included is an example of how one might flatten a netlist in spydrnet.

.. code-block:: python

   import spydrnet as sdn
   from sdn.flatten import flatten

   netlist = sdn.load_example_netlist_by_name(
      'fourBitCounter')

   #flattens in place. netlist will now be flat.
   flatten(netlist)

Uniquify
********

Uniquify is the name we give to the algorithm which helps ensure that each non-terminal instance is unique, meaning that it and it’s definition have a one to one relationship. Non-unique definitions and instances may exist in most netlist formats. One such example could be a four bit adder that is composed of four single bit adders. Assuming that each single bit adder is composed of more than just a single component on the target device, and that the single bit adders are all identical, the design may just define a single single bit adder which it uses in four places. To uniquify this design, new matching definitions for single bit adders would be created for each of the instances of the original single bit adder and the instances that correspond would be pointed to the new copied definitions. Thus each of the definitions would be left with a single instance. 

The uniquify algorithm is very useful when modifications are desired on a specific part of the netlist but not to all instances of the particular component. For example in the four bit adder, if we assume that the highest bit does not need a carry out, the single bit adder there could be simplified. However, if we make modifications to the single bit adder before uniquifying the modifications will apply to all four adders. If we instead uniquify first then we can easily modify only the adder of interest.

Currently :code:`Uniquify` is implemented to ensure that the entire netlist contains only unique definitions. This is one approach to uniquify, however an interesting area for future exploration is that of uniquify on demand. Or some other approach to only ensure and correct uniquification of modified components only. This is left for future work.

The following code example shows uniquify being used in SpyDrNet.

.. code-block:: python

   import spydrnet as sdn
   from sdn.uniquify import uniquify

   netlist = sdn.load_example_netlist_by_name(
      'fourBitCounter')

   uniquify(netlist)


Clone
*****

Cloning is another useful algorithm currently implemented in SpyDrNet. Currently all of the components in a netlist can be cloned from pins and wires to whole netlist objects. Upon initial inspection clone seems simple. However, there is some complexity when it comes to the connections between individual components. Some explanation is provided here.

Clone could be implemented a number of ways. We attempted to find the logical method for our clone algorithm at each level of the data structure. Our overall guiding principles were that at each level, lower level objects should maintain their connections, the cloned object should not belong to any other object, and the cloned object should not maintain its horizontal connections. There are of course some exceptions to these rules which seemed judicious. One such example is that when cloning an instance, That instance will maintain its original corresponding definition, unless the corresponding definition is also being cloned as in the case of cloning a whole library or netlist (in which case the new cloned definition will be used).

Additionally connection modification was done at a level lower than the API in order to mantain consistency as different components were cloned. This promoted code reuse in the clone implementation and helped minimize the number of dictionaries used.

The clone algorithm is very useful while implementing some of the higher level algorithms such as TMR and DWC with compare that we use for reliability research. In these algorithms cloning is essential, and having it built into the tool helps simplify their implementation.

The example code included in this section will clone an element and then add that element back into the netlist which it originally belonged to. Comments are included for most lines in this example to illuminate why each step must be taken. 

.. code-block:: python

   import spydrnet as sdn

   netlist = sdn.load_example_netlist_by_name(
      'hierarchical_luts')

   #index found by printing children's names
   sub = netlist.top_instance.reference.children[2]
   sub_clone = 
      sub.clone()
   
   #renamed needed to be added back into the netlist
   sub_clone.name = "sub_clone"

   #The 'EDIF.identifier' must also be changed 
   #Avoids EDIF namespace plugin naming conflict
   sub_clone["EDIF.identifier"] = "sub_clone"

   #this line adds the cloned instance into the netlist
   netlist.top_instance.reference.add_child(sub_clone)


Hierarchical References
************************

SpyDrNet includes the ability to create a hierarchical reference graph of all of the instances, ports, cables, and other objects which may be instantiated. The goal behind hierarchical references is to create a graph on which other tools, such as NetworkX can more easily build a graph. each hierarchical reference will be unique, even if the underlying component is not unique. These components are also very light weight to minimize memory impact since there can be many of these in flight at one time.

The code below shows how one can get and print hierarchical references. The hierarchical references can represent any spydrnet object that may be instanciated in a hierarchical manner.

.. code-block:: python

   top = netlist.top_instance
   child_instances = top.reference.children

   for h in sdn.get_hinstances(child_instances):
      print(h, type(h.item).__name__)


Getter functions
****************

SpyDrNet includes getter functions which are helpful in the analysis and transformation of netlists. These functions were created to help a user more quickly traverse the netlist. These functions provide the user with quick access to adjacent components. A call to a getter function can get any other related elements from the existing element that the user has a handle to (see Figure :ref:`getterfuncs`). Similar to clone there are multiple methods which could be used to implement a correct getter function. We again strove to apply the most logical and consistent rules for the getter functions. There are some places in which the object returned may not be the only possible object to be returned. In these cases generators are returned. In cases in which there are two possible classes of relationships upon which to return objects, the user may specify whether they would like to get the more inward related or outward related objects. For example, a port may have outer pins on instances or inner pins within the port in the definition. Both of these pins can be obtained separately by passing a flag.

.. figure:: SpyDrNetConnectivity.pdf
   :scale: 100%
   :align: center
   :figclass: htbp

   Getter functions are able to get sets of any element related to any other element. :label:`getterfuncs`

In the example only a few of the possible getter functions are shown. The same pattern can be used to get any type of object from another however. Each call to a getter function returns a generator.

.. code-block::python

   import spydrnet as sdn

   netlist = sdn.load_example_netlist_by_name(
      'fourBitCounter')

   netlist.get_instances()

   netlist.top_instance.get_libraries()

   netlist.top_instance.get_ports()

Applications
------------

SpyDrNet may be used for a wide variety of applications. SpyDrNet grew out of a lab that is focused primarily on 
improving circuit reliability and security.  An application that has had strong influence over its development is that 
of enhancing circuit reliability in harsh radiation environments through partial circuit replication :cite:`pratt_2008`.
When a particle of ionizing radiation passes through an integrated circuit, it can deposit enough energy to invert values 
stored in memory cells :cite:`JEDEC`. An FPGA is a computer chip that can be used to implement 
custom circuits. SRAM-based FPGA stores a circuits configuration in a large array of memory. When radiation corrupts an FPGA 
configuration memory, it can corrupt the underlying circuit and cause failure.

One of our areas of research involves finding ways to design more reliable circuits to be programmed onto existing, non 
specialized, FPGAs. These modifications are useful for designers that deploy many FPGAs as well as designers that plan 
on deploying circuits in high radiation environments where single event upsets can disrupt the normal operation of devices. 
These reliability focused modifications require some analysis of netlist structure as well as modifications in the netlist. 

SpyDrNet was created to help automate this process and allow our researchers to spend more time studying the resulting 
improved circuitry and less time modifying the circuit itself. It is important to note that some care needs to be taken
to ensure that redundancy modifications are not removed by down stream optimizations in implementation. Reliability 
modifications to netlists are often optimized away. One common adjustment to a netlist for reliability purposes, is a 
replication of various components. Often when tools see the same functionality with a theoretical identical result they 
will attempt to remove the duplicated portion and provide two outputs on a single instance. This defeats the purpose of 
the reliability modifications. Using and modifying netlists allows us to bypass those optimizations and gives more 
control over how our design is built. Below are some details on using SpyDrNet for higher level transformation and 
analysis techniques applicable to reliability applications.

Triple Modular Redundancy 
*************************

TMR is one method by which circuits can be made more reliable. TMR triplicates portions of the circuit to allow the circuit to continue to provide the correct result even under some cases of error. Voters are inserted between triplicated circuit components to pass the most common result on to the next stage of the circuit :cite:`pratt_2008`. Figure :ref:`tmrfig` shows two typical layouts for TMR. The top half of the image shows a triplicated circuit with a single voter that feeds into the next stage of the circuit. The bottom of the figure shows a triplicated voter layout such that even a single voter failure may be tolerated.

.. figure:: tmr.png
   :align: center
   :figclass: htbp

   Triple modular redundancy with a single voter and triplicated voters. :cite:`tmrimage` :label:`tmrfig`
   
TMR has been applied using SpyDrNet. The current implementation selects subsets of the circuit to replicate. Then a voter insertion algorithm creates and inserts the voter logic between triplicated layers. Later, reduction voting is added to the output, connecting the triplicated logic in place of the original implementation. The ability of SpyDrNet to carry hierarchy through the tool was taken advantage of by the TMR implementation. This allows the triplicated design to take advantage of the benefits of hierarchy including, improved place and route steps on the target FPGA. Previous work with the BYU EDIF Tools :cite:`BYUediftools` required a flattened design to accomplish TMR on a netlist. The triplicated design was programmed to an FPGA after being processed using SpyDrNet.

Duplication With Compare 
************************

.. figure:: dwc.png
   :align: center
   :figclass: htbp
   
   Duplication with compare showing the duplicated circuitry and duplicated violation flags.


DWC is a reliability algorithm in which the user will duplicate components of the design and include comparators on the output to try present a flag that will be raised when one of the circuits goes down :cite:`johnson_dwc`. Like TMR's voters, the comparators can be duplicated as well to ensure that if a comparator goes down at least one of the comparators will flag an issue.

DWC was again implemented on SpyDrNet. Once again this was able to take advantage of SpyDrNet's hierarchy and maintain that through the build. Comparators were created and inserted and the selected portion of the design was duplicated. The resulting circuits were programmed to an FPGA after being read into SpyDrNet, modified and written back out. As with TMR the existing implementation on the BYU EDIF Tools :cite:`BYUediftools` required that the design be flattened before being processed.

Clock Domain Analysis
*********************

In hardware various clocks are often used in different portions of the circuit. Sometimes inputs and outputs will come in on a different clock before they reach the main pipeline of the circuit. At the junctions between clock domains circutry should not be triplicated in TMR. If it is triplicated it may result in steady state error on the output because the signals from the three inputs may reach the crossing at different times and be registered improperly :cite:`tmr_sync`. This can make the overall reliability of the system lower than it otherwise would be. 

In order to find these locations. Clock domains have been examined using SpyDrNet. The basic methodology for doing this was to find the clock ports on the various components in the design which have them and trace those clocks through the netlist. The resulting connected components form a clock domain. When a triplication pass encountered the boundry between domains the triplicated circuit could be reduced to a single signal to cross the boundry.

Graph Analysis and Feedback
***************************

While triplictaing a design users must determine the best location to insert voters in the design. Voters could be inserted liberally at the cost of the timing of the critical path. Alternatively sparse voter insertion can yield a lower reliability. One consideration to take into account is that voters inserted on feedback loops in the directional graph represented by the netlist can help correct the circuit's state more readily. One study concluded that inserting voters after high fanout flip flops in a design yielded good results. :cite:`Johnson:2010` This voter insertion algorithm was implemented on SpyDrNet after doing analysis using NetworkX :cite:`networkx` to find the feedback loops.

Conclusion
----------

SpyDrNet is a framework created to be as flexible as possible while still meeting the needs of reliability related research. We have worked to ensure that this tool is capable of a wide variety of netlist modifications.

Although this tool is new, a few reliability applications have been built on SpyDrNet. Because of these applications we feel confident that this tool can be helpful to others. SpyDrNet is released on github under an open source licence. New users are welcome to use and contribute to the SpyDrNet tools.

Acknowledgment
--------------

This work was supported by the Utah NASA Space Grant
Consortium and by the I/UCRC Program of the National
Science Foundation under Grant No. 1738550.


