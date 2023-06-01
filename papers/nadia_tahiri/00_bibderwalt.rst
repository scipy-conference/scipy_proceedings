:author: Wanlin Li
:email: Wanlin.Li@USherbrooke.ca
:institution: Department of Computer Science, University of Sherbrooke, Sherbrooke, Canada

:author: Nadia Tahiri
:email: Nadia.Tahiri@USherbrooke.ca
:institution: Department of Computer Science, University of Sherbrooke, Sherbrooke, Canada

:bibliography: mybib


-----------------------------------------------------------------------------------------------------------------------------
aPhyloGeo-Covid: A Web Interface for Reproducible Phylogeographic Analysis of SARS-CoV-2 Variation using Neo4j and Snakemake 
-----------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   The gene sequencing data, along with the associated lineage tracing and research data generated 
   throughout the Coronavirus disease 2019 (COVID-19) pandemic, constitute invaluable resources that profoundly 
   empower phylogeography research. To optimize the utilization of these resources, we have developed a web-based 
   analysis platform called aPhyloGeo-Covid, leveraging the capabilities of Neo4j, Snakemake, and Python. This platform enables users 
   to explore and visualize a wide range of diverse data sources specifically relevant to  SARS-CoV-2 for phylogeographic analysis. 
   The integrated Neo4j database acts as a comprehensive repository, consolidating COVID-19 pandemic-related sequences information, 
   climate data, and demographic data obtained from public databases, facilitating efficient filtering and organization of input data for 
   phylogeographical studies. Presently, the database encompasses over 113,774 nodes and 194,381 relationships. Once the input dataset is determined, 
   aPhyloGeo-Covid provides a scalable and reproducible workflow for investigating the intricate relationship between geographic 
   features and the patterns of variation in different SARS-CoV-2 variants. The platform's codebase is publicly accessible on 
   GitHub (https://github.com/tahiri-lab/iPhyloGeo/tree/iPhylooGeo-neo4j), providing researchers with a valuable tool to analyze 
   and explore the intricate dynamics of SARS-CoV-2 within a phylogeographic context.
   

.. class:: keywords

   Phylogeography, Neo4j, Snakemake, Dash, SARS-CoV-2

Introduction
------------

Phylogeography is a field of study that investigates the geographic distribution of genetic lineages within a particular species, 
including viruses. It combines principles from evolutionary biology and biogeography to understand how genetic variation is distributed 
across different spatial scales :cite:`dellicour2019using`. In the context of viruses, phylogeography seeks to uncover the evolutionary 
history and spread of viral lineages by analyzing their genetic sequences and geographical locations. By examining the genetic diversity 
of viruses collected from various geographic locations, researchers can reconstruct the patterns of viral dispersal and track the movement 
and transmission dynamics of viral populations over time :cite:`vogels2023phylogeographic` :cite:`franzo2022phylodynamic` :cite:`munsey2021phylogeographic`. 
For phylogeographic studies in viruses, researchers typically require integrating genetic sequences, geographic information and 
temporal information. By combining the genetic sequences with geographic information, researchers can analyze the phylogenetic relationships 
among the viral strains and infer the patterns of viral migration and transmission across different regions. By integrating genetic and 
temporal information, researchers can infer the timescale of viral evolution, and trace the origins and dispersal patterns of different viral 
lineages :cite:`holmes2004phylogeography`. Throughout the COVID-19 pandemic, researchers worldwide sequenced the genomes of thousands of SARS-CoV-2 viruses. 
These efforts have helped researchers study the virus's evolution and spread over time and across different geographic regions, which is critical 
to informing public health strategies for controlling future outbreaks. However, the abundance of genetic sequences and the accompanying geographic 
and temporal data are scattered across multiple databases, making it challenging to extract, validate, and integrate the information. For instance, 
to conduct a phylogeographic study in SARS-CoV-2, a researcher would first need access to data on the geographic distribution of specific lineages, 
including the most common countries where they are found, as well as the earliest and latest detected dates. This data is provided by the Cov-Lineages.org 
Lineage Report :cite:`o2021tracking`. Subsequently, based on the most common country and lineage detection dates, the researcher would need to search 
for sequencing data in databases such as NCBI Virus resource :cite:`brister2015ncbi` or GISAID :cite:`khare2021gisaid`. Climate data can be obtained 
from references to datasets like NASA/POWER and DailyGridded weather :cite:`marzouk2021assessment`. Additional data, including epidemiological information 
like COVID-19 testing and vaccination rates, can be retrieved from projects like Our World in Data :cite:`mathieu2021global`. In summary, conducting 
phylogeographic research in viruses involves not only screening and selecting sequencing data but also managing the associated geographic information and 
integrating vast amounts of environmental data. This process can be time-consuming and prone to errors. The challenges associated with data collection, 
extraction, and integration have hindered the advancement of phylogeographic research in the field. To address these challenges, a highly scalable and 
flexible graph database management system Neo4j :cite:`guia2017graph` was applied to store, manage, and query large-scale SARS-CoV-2 variants-related data. 
Unlike traditional relational databases that use tables and rows, Neo4j represents data as a network of interconnected nodes and relationships. 
It leverages graph theory and provides a powerful framework for modelling, storing, and analyzing complex relationships between 
entities :cite:`angles2012comparison`.

On the other hand, while recent phylogeographic studies have extensively analyzed the genetic data of species distributed under different 
geographical locations, many of them have only focused on the distribution of species or provided visual representations without exploring 
the correlation between specific genes (or gene segments) and environmental factors :cite:`uphyrkina2001phylogenetics` :cite:`luo2004phylogeography` 
:cite:`taylor2020intercontinental` :cite:`aziz2022phylogeography`. To fill this gap, a novel algorithm applying sliding windows to scan the genetic 
sequence information related to their climatic conditions was developed by our team :cite:`koshkarov2022phylogeography`. This algorithm utilizes sliding 
windows to scan genetic sequence information in relation to climatic conditions. Multiple sequences are aligned and segmented into numerous alignment windows 
based on predefined window size and step size. To assess the relationship between variation patterns within species and geographic features, the Robinson and 
Foulds metric :cite:`robinson1981comparison` was employed to quantify the dissimilarity between the phylogenetic tree of each window and the topological tree 
of geographic features. However, this process was computationally intensive as each window needed to be processed independently. Additionally, determining 
the optimal sliding window size and step size often required multiple parameter settings to optimize the analysis. Thus, reproducibility played a 
critical role in this process. To address these challenges, we designed a phylogeographic pipeline that leverages Snakemake, a modern computational 
workflow management system :cite:`koster2012snakemake`. Unlike other workflow management systems such as Galaxy :cite:`jalili2020galaxy` and Nextflow 
:cite:`spivsakova2023nextflow`, Snakemake stands out for being written in Python, making it highly portable and requiring only a Python installation to 
run Snakefiles :cite:`wratten2021reproducible`. The Snakemake workflow can harnesses various Python packages, including Biopython :cite:`cock2009biopython` 
and Pandas :cite:`lemenkova2019processing`, enabling efficient handling of sequencing data reading and writing as well as phylogenetic analysis. 
This makes Python-based Snakemake the ideal choice for aPhyloGeo-Covid. Furthermore, the Snakemake pipeline seamlessly integrates with other tools 
through Conda, ensuring efficient dependency and environment management. With a single command, all necessary dependencies can be downloaded and installed. 
Another significant advantage of Snakemake is its scalability, capable of handling large workflows with numerous rules and dependencies. 
It can be executed on various computing environments, including workstations, clusters, and cloud computing platforms like Kubernetes, Google 
Cloud Platform, and Amazon Web Services. Moreover, Snakemake supports parallel execution of jobs, greatly enhancing the pipeline's overall performance and speed.

With these considerations in mind, the main aim of this study is to create an open-source, web-based phylogeographic analysis platform that overcomes 
the aforementioned limitations. This platform comprises two essential components: data pre-processing and phylogeographical analysis. 
In the data pre-processing phase, we utilize searchable graph databases to facilitate rapid exploration and provide a visual overview of 
the SARS-CoV-2 variants and their associated environmental factors. This enables researchers to efficiently navigate through the vast amount of 
data and extract relevant information for their analyses. In the phylogeographical analysis phase, we employ our modularized Snakemake workflow to 
investigate how patterns of genetic variation within different SARS-CoV-2 variants align with geographic features. By utilizing this workflow, 
researchers can analyze the relationship between viral genetic diversity and specific geographic factors in a structured and reproducible manner. 
This comprehensive approach allows for a deeper understanding of the complex interplay between viral evolution, transmission dynamics, 
and environmental influences.

Methodology
-----------
Various data sources related to SARS-CoV-2 were extracted, transformed, and loaded into a Neo4j graph database. To facilitate querying, analysis parameter configuration, and output writing within the database, a driver object was created using the Neo4j Python driver to establish connections with the Neo4j database. For phylogeographic analysis, a workflow was implemented using the Snakemake workflow management system, ensuring a streamlined and efficient analysis process. Additionally, the Plotly-dash library and its interactive visualization tools were utilized for exploring input data, setting analysis parameters, and visualizing the results in an interactive manner.


Neo4j graph database and Dash platform
---------------------------------------

A graph database is a type of database management system (DBMS) that uses graph structures for data representation and query processing :cite:`timon2021overview`. 
Unlike traditional relational databases that store data in tables with rows and columns, graph databases organize data as nodes, edges, and properties. In a graph database, nodes represent entities or objects, edges represent the relationships between nodes, and properties provide additional information about nodes and edges. One of the critical advantages of graph databases is their ability to traverse and query interconnected data efficiently. Graph databases excel at handling queries involving relationship patterns, graph algorithms, and path traversals. They enable efficient navigation through complex networks, enabling robust graph-based analyses and insights :cite:`vicknair2010comparison`.

Data Integration 
++++++++++++++++

Various data sources related to SARS-CoV-2 were integrated into a Neo4j database, covering the period from January 1, 2020, to December 31, 2022. The data sources include SARS-CoV-2 sequences from the SARS-CoV-2 Data Hub :cite:`brister2015ncbi`, lineage development information from Cov-Lineages :cite:`o2021tracking`, population density by country, positivity rates, vaccination rates, diabetes rates, aging data from Our World in Data :cite:`mathieu2021global`, and climate data from NASA/POWER :cite:`marzouk2021assessment`. Within the Neo4j database, we defined several labels to organize the data. These labels include Lineage, Protein, Nucleotide, Location, and LocationDay (See :ref:`fig1`). The Protein and Nucleotide labels store sequencing data information such as Accession, length, collection date, and collected country. The Lineage label stores lineage development information, including the most common country, latest date, and earliest date associated with each lineage. The LocationDay label stores climate information such as temperature, precipitation, wind speed, humidity and sky shortwave irradiance for each location and specific day. The Location label contains basic information about hospitals, health, and the economy of each country, including GDP, median age, life expectancy, population, the proportion of people aged 65 and older, proportion of smokers, proportion of extreme poverty, diabetes prevalence, human development index, and more. Lineage nodes are connected to Nucleotide and Protein nodes, representing the relationships between lineages and their associated genetic sequence data. Lineage nodes also have relationships with Location nodes, using the most common occurrence rate as a property. This design allows users to determine the most common countries based on lineage names or search for lineages that were most common in specific countries during a certain time period.


.. figure:: figure1.png

   Schema of Neo4j Database for Phylogeographic Analysis of SARS-CoV-2 Variation. The schema includes key entities and relationships essential for organizing and querying data related to samples of protein, samples of nucleotide, locations, lineages, analysis input, output and parameters. Each entity represents a distinct aspect of the analysis process and facilitates efficient data organization and retrieval. :label:`fig1`


Input exploration
+++++++++++++++++

To provide users with an interactive interface, we developed a web-based platform using Dash-Plotly :cite:`liermann2021dynamic`. Connecting the Dash Web platform to the Neo4j graph database enables quick retrieval of relevant data information from related nodes when users provide keywords about lineages or locations. This functionality allows users to quickly identify and filter the appropriate datasets for further phylogeographic analysis. By combining the power of the Neo4j database and the user-friendly web-based platform, our design facilitates efficient data exploration and selection, supporting researchers in their phylogeographic analysis of SARS-CoV-2 variation.

The aPhyloGeo-Covid provids two approaches to select input datasets.

1. Determine the most common country for the lineages based on the name of the lineage, and then retrieve the corresponding sequences.

The multi-step process is facilitated by the "Neo4j GraphDatabase" Python package :cite:`jordan2014neo4j` and the interactive Dash web page. Firstly, users select specific lineages of interest from a checklist on the Dash web page. Next, utilizing the capabilities of the "Neo4j GraphDatabase" package, the selected lineages are used to query the graph database, retrieving relevant location information such as associated locations, earliest and latest detected dates of the lineages in the most common location, and their most common rates. Once these results are obtained from the database, they are presented on the web page as an interactive Dash Table. This table provides a user-friendly interface, allowing users to apply columns and rows filters. This feature enables the removal of study areas or lineages deemed irrelevant, as well as excluding lineages with a most common rate below a predetermined threshold. Finally, based on the filtered table and the selected sequence type, the "Neo4j GraphDatabase" package extracts all the related sequences by accession number. These filtered sequences were then collected as part of the input data for subsequent phylogeographic analysis.

The following "update_lineage_table" function serves as a callback in the application and is triggered when the user clicks the "button-confir-lineage" component. Its purpose is to update the lineage table based on the selected values from the "choice-lineage" and "type-dropdown" components. Upon invocation, the function checks if the callback was triggered and verifies the presence of valid checklist values and sequence type. If these conditions are met, a Cypher query is constructed to retrieve the most common country information associated with the specified lineage. The query results are then processed and transformed into a pandas DataFrame. Further data manipulation can be performed at this stage. Finally, the resulting DataFrame is converted into a list of dictionaries, which serves as the updated data for the lineage table component. By utilizing this function, the application effectively enables users to explore and visualize lineage data in the context of geographic features
      

.. code-block:: python

   @ app.callback(
       Output('lineage-table', 'data'),
       Output('valid-message', 'children'),
       Input('button-confir-lineage', 'n_clicks'),
       State('choice-lineage', 'value'),
       State('type-dropdown', 'value')
   )

   def update_lineage_table(n, checklist_value, seqType_value):
       if n is None:
           return None, None
       else:
           if checklist_value and seqType_value:
           # Query most common country in Neo4j database based on the name of the lineage
               starts_with_conditions = " OR ".join(
                   [f'n.lineage STARTS WITH "{char}"' for char in checklist_value])
               query = f"""
                   MATCH (n:Lineage) - [r: IN_MOST_COMMON_COUNTRY] -> (l: Location)
                   WHERE {starts_with_conditions}
                   RETURN n.lineage as lineage, n.earliest_date as earliest_date, 
                     n.latest_date as latest_date, l.iso_code as iso_code, 
                     n.most_common_country as most_common_country,  r.rate as rate
                   """
               cols = ['lineage', 'earliest_date', 'latest_date', 'iso_code',
                       'most_common_country', 'rate']
                # Transform Cypher results to pandas dataframe
               df = neoCypher_manager.queryToDataframe(query, cols)
               ...
               table_data = df.to_dict('records')
               return table_data, None
           ....

2. Search for lineages that were most common in a specific country during a certain time period, and then retrieve the corresponding sequences.

This approach involved users defining specific locations and a date period through the Dash web page. Utilizing the capabilities of the GraphDatabase package, the Neo4j database is queried to identify lineages prevalent in the specified locations during the defined time period. The retrieved information includes the earliest and latest detected dates of the lineages in each country and their most common rates. These results were presented to users through an interactive Dash Table, which facilitated the application of filters to eliminate outside study areas or lineages below a predetermined threshold. Then, the GraphDatabase package is utilized again to filter and extract the accession number of the corresponding sequences, which are then collected for subsequent phylogeographic analysis.

In the following code, the "update_table function" is a callback in the application that responds to the user clicking the "button-confir-lineage2" component. Its purpose is to update the location table based on the selected start and end dates, checklist values, and sequence type. The function constructs a Cypher query to retrieve lineage data from the Neo4j database, filtering it based on the specified location and date criteria. The query results are transformed into a pandas DataFrame and further data manipulation can be performed. The resulting DataFrame is converted to a list of dictionaries, which serves as the updated data for the location table. By utilizing this function, the application enables users to explore and visualize lineage data associated with different geographic regions within a specified date range, facilitating the study of phylogeographic patterns and variations.
   

.. code-block:: python

   @ app.callback(
       Output('location-table', 'data'),
       Output('valid-message2', 'children'),
       Input('button-confir-lineage2', 'n_clicks'),
       State('date-range-lineage', 'start_date'),
       State('date-range-lineage', 'end_date'),
       State('choice-location', 'value'),
       State('type-dropdown2', 'value')
   )
   def update_table(n, start_date_string, end_date_string, checklist_value, seqType_value):
       if n is None:
           return None, None
       else:
           if start_date_string and end_date_string and checklist_value and seqType_value:
           # Query lineage data in Neo4j database based on the name of location and date  
               start_date = datetime.strptime(
                   start_date_string, '%Y-%m-%d').date()
               end_date = datetime.strptime(
                   end_date_string, '%Y-%m-%d').date()
               query = f"""
                   MATCH (n:Lineage) - [r: IN_MOST_COMMON_COUNTRY] -> (l: Location)
                   WHERE n.earliest_date > datetime("{start_date.isoformat()}") 
                        AND n.earliest_date < datetime("{end_date.isoformat()}")
                   AND l.location in {checklist_value}
                   RETURN n.lineage as lineage, n.earliest_date as earliest_date, 
                           n.latest_date as latest_date, l.iso_code, 
                        l.location as most_common_country,  r.rate
                   """
               cols = ['lineage', 'earliest_date', 'latest_date', 'iso_code',
                       'most_common_country', 'rate']
               # Transform Cypher results to pandas dataframe
               df = neoCypher_manager.queryToDataframe(query, cols)
               # Convert the 'Date' column to pandas datetime format
               ...
               table_data = df.to_dict('records')
               return table_data, None
           ...


In summary, these approaches leveraged the "Neo4j GraphDatabase" package and the interactive Dash web page to enable user-driven sequencing searching. Once input sequencing has been defined, an Input node is generated and labelled accordingly in our graph database. 
This Input node is connected to each sequencing (Nucleotide or Protein) node used in the analysis, establishing relationships between the input data and the corresponding sequences. Each Input node is assigned a unique ID, which is provided to the client for reference.

The following functions facilitate the generation of unique names for nodes and the addition of input nodes with relationships to other nodes in the Neo4j database, contributing to the organization and management of data in a scientific context. 
The "generate_unique_name" function generates a unique name for a node in the Neo4j database. It takes the label of the node as input and uses a randomly generated short ID to create a unique name. It utilizes a Neo4j driver to establish a connection with the database, checks if a node with the generated name already exists, and continues generating a new name until a unique one is found. The function returns the unique name.
The "addInputNeo" function adds an input node and establishes relationships with other nodes in the Neo4j database. It takes the label of the nodes, the name of the input node, and a list of IDs as input. It uses a Neo4j driver to connect to the database and creates a new input node with the specified name. It then performs a MATCH query to retrieve nodes with IDs present in the provided list. For each matched node, a relationship of type "IN_INPUT" is created between the input node and the matched node. The function prints a message to indicate that an input node has been successfully added to the Neo4j database.

.. code-block:: python

   def generate_unique_name(nodesLabel):
       driver = GraphDatabase.driver(URI,
                                     auth=("neo4j", password))
       with driver.session() as session:
           random_name = generate_short_id()

           result = session.run(
               "MATCH (u:" + nodesLabel + " {name: $name}) RETURN COUNT(u)", name=random_name)
           count = result.single()[0]

           while count > 0:
               random_name = generate_short_id()
               result = session.run(
                   "MATCH (u:" + nodesLabel + " {name: $name}) RETURN COUNT(u)", name=random_name)
               count = result.single()[0]

           return random_name

   def addInputNeo(nodesLabel, inputNode_name, id_list):
       # Execute the Cypher query
       driver = GraphDatabase.driver(URI,
                                     auth=("neo4j", password))

       # Create a new node for the user
       with driver.session() as session:
           session.run(
               "CREATE (userInput:Input {name: $name})", name=inputNode_name)
       # Perform MATCH query to retrieve nodes
       with driver.session() as session:
           result = session.run(
               "MATCH (n:" + nodesLabel + ") WHERE n.accession IN $id_lt RETURN n",
               nodesLabel=nodesLabel,
               id_lt=id_list)
           # Create relationship with properties for each matched node
           with driver.session() as session:
               for record in result:
                   other_node = record["n"]
                   session.run("MATCH (u:Input {name: $name}), 
                                 (n:" + nodesLabel + " {accession: $id}) "
                               "CREATE (n)-[r:IN_INPUT]->(u)",
                               name=inputNode_name, 
                               nodesLabel=nodesLabel, 
                               id=other_node["accession"])
       print("An Input Node has been Added in Neo4j Database!")


Parameters setting and tuning
++++++++++++++++++++++++++++++

Once the input data has been defined, including sequence data and associated location information, the platform guides users to select the parameters for analysis. At this step, a Label named Analysis is created, and the values of the parameters are saved in the node as properties. These parameters include step size, window size, RF distance threshold, bootstrap threshold, and the list of the environmental factors involved in the analysis. Then a connection between the Input Node and the Analysis Node is created, which offers several advantages. Firstly, it enables users to compare the differences in results obtained from the same input samples but with different parameter settings. Secondly, it facilitates the comparison of analysis results obtained using the same parameter settings but different input samples. The networks of Input, Analysis, and Output nodes (:ref:`fig1`) ensure repeatability and comparability of the analysis results.

The following "create_Analysisnode" function creates an Analysis node in the Neo4j database by executing a Cypher query with the provided data. On the other hand, the "addAnalysisNeo" function adds an Analysis node to the Neo4j database, sets its properties using values from a YAML file, and establishes a relationship between the Analysis node and an existing Input node. The functions utilize a Neo4j driver to connect to the database and perform the necessary operations. By using these functions together, researchers can easily create Analysis nodes with customized properties and establish connections to relevant Input nodes in the Neo4j database, facilitating the management and analysis of scientific data.


.. code-block:: python

   def create_Analysisnode(tx, data):
       query = "CREATE (n:Analysis) SET n = $data"
       tx.run(query, data=data)

   def addAnalysisNeo():
       driver = GraphDatabase.driver(URI,
                                     auth=("neo4j", password))

       properties_dict = {}
       with open('config/config.yaml', 'r') as file:
           config = yaml.safe_load(file)

       # Set the properties of the node using the yaml_data
       set_properties(config, properties_dict)
       input_name = properties_dict['input_name']
       analysis_name = properties_dict['analysis_name']
       create_time = datetime.now().isoformat()
       # Create node
       with driver.session() as session:
           session.execute_write(create_Analysisnode, properties_dict)
       # Create relationship
       with driver.session() as session:
           session.run("MATCH (u:Input {name: $input_name}), (n:Analysis {analysis_name: $analysis_name}) "
                       "CREATE (u)-[r:FOR_ANALYSIS {create_time: $create_time}]->(n)",
                       input_name=input_name, analysis_name=analysis_name, create_time=create_time)
       print("An Analysis Node has been Added in Neo4j Database!")


Subsequently, when the user confirms the start of the analysis with the SUBMIT button, the corresponding sequences are downloaded from NCBI :cite:`brister2015ncbi` using the Biopython package :cite:`cock2009biopython`, and multiple sequence alignments (MSA) :cite:`edgar2006multiple` are performed using the MAFFT method :cite:`katoh2013mafft`. With alignment results and related environmental data as input, the Snakemake workflow will be triggered in the backend. Once the analysis is completed, the user is assigned a unique output ID, which they can use to query and visualize the results in the web platform.
   


Output exploration
++++++++++++++++++

At the end of each analysis, an output node with a unique id is created in the Neo4j graph database. The associated nodes containing input and parameter information are connected to it by edges. Therefore, users can retrieve and visualize the analysis results through Output ID. The platform allows users to query individual results but also provides the capability to compare the results of multiple analyses. 

Input, Analysis, and Output nodes created by different users form a network that encompasses numerous combinations of parameter settings and input configurations. As the utilization of the platform expands, this network grows, resulting in an open academic platform that fosters communication and collaboration. This feature enhances the user's ability to gain insights from the data and enables comprehensive analysis of the phylogeographic patterns of SARS-CoV-2 variation.


Snakemake workflow for phylogenetic analysis
---------------------------------------------

In this study, a combination of sliding window strategy and phylogenetic analyses was used to explore the potential correlation between the diversity of specific genes or gene fragments and their geographic distribution. The approach involved partitioning a multiple sequence alignment into windows based on sliding window size and step size. Phylogenetic trees were constructed for each window, and cluster analyses were performed for various geographic factors using distance matrices and the Neighbor-Joining clustering method :cite:`mihaescu2009neighbor`. The correlation between phylogenetic and reference trees was evaluated using Robinson and Foulds (RF) distance calculation. Bootstrap and RF thresholds were applied to identify gene fragments with variation patterns within species that coincided with specific geographic features, providing informative reference points for future studies. The workflow encompassed steps such as reference tree construction, sliding windows, phylogenetic tree construction, preliminary filtering based on bootstrap threshold and RF distance, advanced phylogenetic tree construction, and further filtering based on bootstrap threshold and RF distance. The workflow utilized tools and software like Biopython :cite:`cock2009biopython`, raxml-ng :cite:`kozlov2019raxml`, fasttree :cite:`price2009fasttree`, and Python libraries such as robinson-foulds, NumPy, and pandas for data parsing, phylogenetic inference, RF distance calculation, mutation testing, and filter creation. A manuscript for aPhyloGeo-pipeline is available on Github Wiki (https://github.com/tahiri-lab/aPhyloGeo-pipeline/wiki).
   

Results
-------

The SARS-CoV-2 virus has a genome size of approximately 30kb. The first two-thirds of its genome, located at the 5'-terminal, contain instructions for producing two long proteins known as pp1a and pp1ab. These proteins are subsequently broken down by viral enzymes into 16 smaller proteins called non-structural proteins (Nsps), which are important for the virus to replicate and transcribe its genetic material :cite:`khan2021structures`. Consequently, the performance of the aPhyloGeo-Covid platform was tested around pp1a. 
To identify and filter the appropriate datasets for further phylogeographic analysis around pp1a, 14 lineages starting with the codes AE, AY, B, BA, BE, DL, or XBB were selected from the checklist on the aPhyloGeo-Covid web page. Subsequently, with the Neo4j graph database, eight relevant locations were retrieved, where at least one of selected lineage was most prevalent (:ref:`fig2`). Based on the lineages, their respective countries, the earliest detected date, and the corresponding protein sequences, input node was created. Certain parameters were defined using the aPhyloGeo-Covid web page, including the step size (3 residues), window size (100 residues), RF distance threshold (100%), bootstrap threshold (0%), and a list of climate factors (humidity, wind speed, sky shortwave irradiance, and precipitation). These parameters were associated with a label called "Analysis" and stored as properties in the node. Finally, the Snakemake workflow would be triggered in the backend. At the conclusion of analysis, an output node with a unique identifier would be generated within the Neo4j graph database (:ref:`fig2`).

.. figure:: figure2.png

   TO BE CONTINUE :label:`fig2`

   
.. figure:: figure3.png

   TO BE CONTINUE :label:`fig3`



Conclusions and future work
---------------------------

To further improve aPhyloGeo-Covid, several potential directions can be considered:

1.	Enhancing Data Resources: To enhance data resources and provide a more comprehensive understanding of the spatial and environmental factors influencing the spread and evolution of SARS-CoV-2, it is crucial to expand the available data resources, especially geographic and environmental data. This expansion may involve the incorporation of additional datasets, such as epidemiological information. By increasing the richness and diversity of data, the aPhyloGeo-Covid project can yield valuable insights into the dynamics of SARS-CoV-2 transmission and its relationship with geographical and environmental variables.

2.	Expanding Phylogeographic Analysis Workflows: In order to enhance the scope of phylogeographic analysis and investigate the evolutionary dynamics and spatial spread of the virus in a more comprehensive manner, it is advisable to expand the existing pipeline of aPhyloGeo-Covid. Alongside the current pipeline, which explores the correlation between specific genes or gene fragments and their geographic distribution, it is recommended to incorporate additional phylogeographic analysis workflows. By incorporating a broader range of analysis approaches, aPhyloGeo-Covid can provide a more extensive toolkit for studying the evolutionary dynamics and spatial dissemination of SARS-CoV-2. This expanded toolkit will contribute to a more comprehensive understanding of the virus and its transmission patterns.

3.	Scalability and Efficiency: To ensure the platform's capability to handle growing user demands and accommodate larger datasets, it is essential to prioritize scalability and efficiency in the development of aPhyloGeo-Covid. Improving scalability and efficiency will result in a platform that can handle larger volumes of data without compromising performance. This will enable researchers and public health practitioners to obtain rapid and reliable analyses, even as the data volumes continue to grow. By ensuring scalability and efficiency, aPhyloGeo-Covid can effectively support decision-making processes and provide valuable insights into the spatial spread and evolution of SARS-CoV-2.



Acknowledgements
----------------

The authors thank SciPy conference and reviewers for their valuable comments on this paper. 
This work was supported by the Natural Sciences and Engineering Research Council of Canada, the Université de Sherbrooke grant, 
and the Centre de recherche en écologie de l’Université de Sherbrooke (CREUS).

