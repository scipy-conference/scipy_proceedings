---
title: A Reproducible Data Lakehouse for High-Resolution Gastric Cancer Epidemiology Study in Chile
abstract: |
  Cancer is the main cause of mortality in Chile since 2019. National surveillance of cancer incidence, often requires merging massive and heterogeneous datasets to generate high resolution insights. For a PhD researcher in Public Health set to understand the spatio-temporal trends of cancer incidence, the lack of high-performance server infrastructure and limitations of traditional epidemiological tools when handling longitudinal Big Data on local machines can act as substantial barriers.

  We developed a portable, open-source Data Lakehouse architecture built entirely within the Python ecosystem to study gastric cancer incidence in the Chilean population. For this purpose, we used anonymized longitudinal data, over a 21 years and 20 million inhabitants including mortality records, hospital discharges, and insurance claims by age, sex, and county, which were stored using Parquet files and Google Drive provided by the University of Chile. We implemented a modular ELT workflow using Jupyter Notebooks and Google Colab, where each notebook was designed to perform a specific extraction, normalization or loading process for each dataset. In addition, we employed `rpy2` and `mgvc` packages for age-standardized time-series modeling, GeoPandas and Sci-kit for spatial analysis and identification of high-risk zones; and NetworkX to model patient-consultation trajectories, identifying gastroenterology referral clusters in the national health system. Finally, we developed a Streamlit dashboard that allows team members and the general public to visualize cancer incidence time-series charts and referrals clusters dynamically.

  This architecture served as the analytical backbone for a PhD thesis, supporting the submission of three conference abstracts and three manuscripts to a public health journals, ensuring fully reproducible and documented results. This project demonstrates that high-resolution, national-level public health research does not require expensive enterprise software and can be enabled by the production of public health data infrastructure. By utilizing a version-controlled, cloud-collaborative, and nearly zero-cost Python stack, we provide a blueprint for researchers in resource-limited settings to conduct reproducible epidemiological surveillance.

---

## Introduction

Cancer is a priority health problem in Chile. It is currently the leading cause of death and the main burden of disease in the country [@doi:10.4067/S0034-98872021000100149]. Gastric cancer (GC) is particularly prevalent, with Chile having one of the highest incidence and mortality rates in the world alongside with some East Asian countries [@ferlay_j_global_2024].

Chile maintains population-based cancer registries (PCRs) that collect information on all cancer cases across a set of regions for epidemiological surveillance and planning cancer control policies. These registries cover approximately 20% of the national population; however, estimates are reported at the national level, and updated official figures are sparsely reported (not been available since 2019).

We set out to update estimates of GC incidence at the provincial level to support the planning and evaluation of prevention policies. This required applying state-of-the-art methodologies, which demand a large amount of longitudinal data on mortality, hospital discharges, population by sex and age, and socioeconomic variables.

Handling this data with programming languages typically used in epidemiology, such as Stata and R, posed a significant computational risk. Public health researchers often conduct studies using the same hardware used for office tasks, as they use statistics aggregated at the national level. For higher geographic granularity and longer period studies, as in this case, high-performance computing was necessary but not available. Furthermore, tools like R and RStudio use in-memory processing, meaning the entire dataset must be loaded into the computer's RAM for analysis. When handling two decades of longitudinal data for a population of 20 million inhabitants—including heterogeneous records such as mortality, hospital discharges, and insurance claims—the resulting datasets can easily exceed dozens of gigabytes. Attempting to merge and normalize these massive datasets on a typical laptop or office computer could have led to system crashes or "out of memory" errors.

The aim of this study was to develop a portable, open-source "Data Lakehouse" architecture to study Gastric Cancer incidence in Chile's population. Instead of a local, monolithic approach, we moved to a modular, cloud-native architecture that enabled analysis to run from any computer.

## Methods

### Data Architecture

We implemented a Data Lakehouse architecture to bridge the gap between the flexible, cost-effective storage of a data lake and the rigorous, analytical structure of a data warehouse [@zeeb_health_2025]. This hybrid approach allowed us to store 21 years of longitudinal population data in a central repository before subjecting it to specific transformation processes. We utilized an Extract-Load-Transform (ELT) workflow, loading raw datasets with basic normalization into our lake before transforming them into multidimensional structures optimized for epidemiological analysis.

To manage the complexity of transforming heterogeneous health records, we adopted the Dynamic ETL (D-ETL) framework, an approach that combines automatization of the transformation with manual specifications [@ong_dynamic-etl_2017]. Traditional GUI-based ETL tools often lack the flexibility required for the complex requirements of health data and suffer from a lack of transparency in their underlying transformation logic. Following this approach, we modularized our data processing by mapping and transforming each dataset (e.g., mortality records, hospital discharges) separately, allowing us to automate repetitive tasks using scalable, reusable, and customizable Python code while retaining manual control over complex mappings for epidemiological analysis.

### Workflow Implementation

We took advantage of University of Chile´s Google Workspace to use low-cost cloud storage in Google Drive. This also ensured that data storage complied with the University's privacy and security policies. The workflow was arranged in three main folders: 1) Original Data, for the original files (zip, xls, csvs, etc) and documentation received and organized in subfolders by data source, 2) Raw Data, for the data extracted into optimized format for intake, and 3) Data Warehouse, for normalized and standardized data. Additionally, we created a separate folder for each analysis, containing the data transformation script and the statistical analysis.

Jupyter Notebook, a friendly and open source web platform [@jupyter], was used as the coding environment and enabled us to combine data-mapping information in text cells with Python code to generate each target dataset. For each analysis, we created a multidimensional table (cube) and its corresponding Jupyter notebook. All notebooks used Pandas library [@pandas1; @pandas2] and were generated and executed using Google Colab, which connected to Drive folders to load and store datasets and ran on Google servers. Since we used the free version, we had a limit of 12 GB RAM and 50 GB of memory per session.

We used Parquet format to store all datasets due to its high performance for storage and loading. Also, it allowed for storing text metadata, such as text encoding, which is often problematic in Spanish due to special characters. Figure 1 shows the tools used.

:::{figure} figure1.png
:label: fig:process-diagram
Tools used to implement Dynamic Extraction-Load-Transform workflow.
:::

### Data model and sources

We used an Effective Coverage framework that distinguishes populations, needs, resources, production and health outcomes [@marsh_effective_2020] and the Chilean Norm for Health Information for data modeling [@departamento_de_estadistica_e_informacion_de_salud_norma_2016; @departamento_de_estadisticas_e_informacion_de_salud_norma_2023]. @fig:entity-relationship-diagram shows the entities, attributes and relationships used to model GC prevention that were stored in the data lake. @fig:star-schema shows the star-schema for the data cube (multidimensional table) used to analyze GC incidence.


:::{figure} figure2.png
:label: fig:entity-relationship-diagram
Entity-Relationship Diagram.
Colors: blue=populations; yellow=resources; brown=production; green=needs; red=health outcomes.
*Individual-level data.
**Linkable individual-level data.
:::

:::{figure} figure3.png
:label: fig:star-schema
Star schema of data cube used to analyze gastric cancer incidence in Chile 2003-2024.
:::

We retrieved GC cases (ICD-10 category "C16") from the PBCR database (1998–2019) via the Ministry of Health website. Upon request, the Department of Health Statistics and Information (DEIS) provided individual-level records for GC deaths and hospital discharges (2003–2024). We obtained age- and sex-specific population projections and rurality data from the National Statistics Institute (INE). Finally, we sourced insurance coverage data from public health insurer FONASA (2003–2024), and from the Superintendency of Health for ISAPRE beneficiaries (2010–2024); poverty percentages from the Ministry of Social Development; and consultations and UGE production in public hospitals from monthly statistical reports from DEIS (2009-2024).

### Statistical methods and packages

To estimate GC incidence, we used mortality-incidence ratios [@chatignoux_how_2021] implemented with Quasi-Poisson regression models with R's `mgcv` library [@wood_mgcv_2000]. We embedded R scripts in Python code using the `rpy2` library [@gautier_rpy2_2023]. We used `NetworkX` to generate health center networks for GC diagnosis, based on referral patterns [@hagberg_exploring_2008]. We used `Geopandas` for maps [@kelsey_jordahl_2020_3946761] and `Streamlit` to publish a dashboard [@khorasani_web_2022,].

## Results

Table 1 shows the Jupyter notebooks developed. Since each original dataset had a unique script associated, notebooks were stored in the Original Data folder, in their corresponding subfolder (figure 1). During the loading stage, each normalized entity had its own script, so notebooks were stored in the Data Warehouse folder.

```{list-table} Jupyter notebooks developed.
:label: tbl:materials
:header-rows: 1
* - Original Data *Extraction*
  - *Loading* to Data Warehouse
  - *Transformation* Multidimensional Tables
* - Population estimates, Poverty estimates, FONASA Beneficiaries, ISAPRE Beneficiaries, Health center registered population, Health centers, Production, Claims, Referrals, Population-Based Cancer Registry, Hospital discharges, Deaths, Administrative maps
  - District Population, FONASA Beneficiaries, ISAPRE Beneficiaries, Health Center Population, Health centers, Consultation production, Endoscopy production, Cancer referrals, Non-cancer referrals, Gastric cancer cases, Hospital discharges, Deaths, DIM Age, DIM Sex, DIM Districts, DIM Cancer, GEO Provinces, GEO Health services.
  - Population Based Cancer Registry cube, Hospital discharges and deaths events, Gastric cancer cube, Health centers cube, Registered population cube, Referrals fact, Diagnosis coverage cube
```

DIM tables were dimensions common to all analyses used to optimize data normalization and integration. For instance, in `DIM_Age`, for each age `[0, 1, …, 100]`, age ranges of 5 years, 10 years, and other custom ranges are specified. This facilitated the normalization and integration of datasets with different age ranges.


:::{table} Data Lakehouse tables, columns and size.
:label: tbl:areas-html
<table border="1">
  <thead>
    <tr>
      <th>Entity type</th>
      <th>Table</th>
      <th>Columns</th>
      <th>Entries</th>
      <th>Colab Memory Usage</th>
      <th>File size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Populations</td>
      <td>DW_POBLACION_COMUNA</td>
      <td>Comuna, Sexo, Edad, Poblacion, Año, Porc_rural_comuna</td>
      <td>1905768</td>
      <td>87.2 MB</td>
      <td>3.2 MB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_BENEFICIARIOS_FONASA</td>
      <td>Año, Comuna, Sexo, RangoEdad10, Tramo, Beneficiarios</td>
      <td>580445</td>
      <td>32.1 MB</td>
      <td>3.6 MB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_BENEFICIARIOS_ISAPRE</td>
      <td>Año, Sexo, Comuna, RangoEdad5, Beneficiarios</td>
      <td>270572</td>
      <td>12.4 MB</td>
      <td>2.3 MB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_INSCRITOS</td>
      <td>Año, Id_establecimiento, Comuna Id, Sexo, Edad, N_inscritos</td>
      <td>186100</td>
      <td>9.9 MB</td>
      <td>424 KB</td>
    </tr>
    <tr>
      <td>Needs</td>
      <td>DW_SIC_GES</td>
      <td>Año, Mes, Origen, Destino, Sexo, RangoEdad5, Presta_min, Desc_garantia, N_Problema_Salud, Cod_Beneficiario</td>
      <td>200236</td>
      <td>15.3 MB</td>
      <td>1.9 MB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_SIC_SIGTE</td>
      <td>Sexo, Presta_min, Especialidad, Origen, Destino, C_SALIDA, Archivo, Año, Mes, Edad, SIC</td>
      <td>501832</td>
      <td>42.1 MB</td>
      <td>881 KB</td>
    </tr>
    <tr>
      <td>Production</td>
      <td>DW_CENTROS</td>
      <td>Codigo, Nombre, Comuna, Servicio, Servicio_nombre, Region, Nivel, Tipo, Complejidad, Latitud, Longitud</td>
      <td>4582</td>
      <td>393.9 KB</td>
      <td>205 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_PRODUCCION_REM_CONSULTAS</td>
      <td>Año, Mes, Codigo, Prestacion, Produccion, ProduccionHombres, ProduccionMujeres, ProduccionCNE, ProduccionCNE_TM, ProduccionTM, nsp, nsp_cne</td>
      <td>168150</td>
      <td>15.4 MB</td>
      <td>493 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_PRODUCCION_REM_PROCEDIMIENTOS</td>
      <td>Codigo, Prestacion, Año, Mes, Produccion</td>
      <td>88749</td>
      <td>4.1 MB</td>
      <td>728 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_ENDOSCOPIAS_MLE</td>
      <td>Prestacion, Comuna, RangoEdad10, Sexo, Año, Bonos, BonosEst</td>
      <td>163626</td>
      <td>10.5 MB</td>
      <td>1.5 MB</td>
    </tr>
    <tr>
      <td>Health Outcomes</td>
      <td>DW_CASOS_RPC</td>
      <td>idComuna, Sexo, Edad, Mes_diag, Año_diag, CIE10, RPC, Ncasos</td>
      <td>79705</td>
      <td>5.2 MB</td>
      <td>4.5 MB (pkl file)</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_EGRESOS_CA</td>
      <td>idPersona, Sexo, Edad, Provincia, Comuna, Prevision, Año, CIE10, CondicionEgreso, CodigoEst</td>
      <td>599288</td>
      <td>46.3 MB</td>
      <td>35.8 MB</td>
    </tr>
    <tr>
      <td></td>
      <td>DW_DEFUNCIONES_CA</td>
      <td>idPersona, Año, Sexo, Edad, Provincia, Comuna, CIE10, idLugarDefuncion, LugarDefuncion</td>
      <td>210994</td>
      <td>14.5 MB</td>
      <td>14.1 MB</td>
    </tr>
    <tr>
      <td>Dimensions</td>
      <td>DIM_EDAD</td>
      <td>Edad, RangoEdad5, RangoEdad10, RangoEdad20, RangoEdad40, RangoEdad4080</td>
      <td>101</td>
      <td>12.8 KB</td>
      <td>12 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>DIM_SEXO</td>
      <td>Codigo, Sexo, Sexo 2</td>
      <td>5</td>
      <td>252 bytes</td>
      <td>3 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>DIM_COMUNAS</td>
      <td>Comuna, Nombre Comuna, Provincia, Nombre Provincia, idServicio, Servicio, Region, Nombre Region, Macrorregion</td>
      <td>346</td>
      <td>24.5 KB</td>
      <td>14 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>DIM_CANCER</td>
      <td>CIE10, Cancer, Categoria, Digestivo</td>
      <td>412</td>
      <td>13 KB</td>
      <td>9 KB</td>
    </tr>
    <tr>
      <td>Geometries</td>
      <td>GEO_PROVINCIAS</td>
      <td>Provincia, geometry</td>
      <td>56</td>
      <td>1.0 KB</td>
      <td>70.6 MB</td>
    </tr>
    <tr>
      <td></td>
      <td>GEO_SERVICIOS</td>
      <td>Servicio, geometry</td>
      <td>29</td>
      <td>596.0 bytes</td>
      <td>76.5 MB</td>
    </tr>
    <tr>
      <td>Cubes</td>
      <td>CUBO_incidencia</td>
      <td>idComuna, Año, RPC_nombre, RPC, Sexo, RangoEdad, Nombre_Cancer, Nombre Comuna, Poblacion, Ncasos, Ndefunciones</td>
      <td>120240</td>
      <td>11.0 MB</td>
      <td>4.8 MB (xlsx file)</td>
    </tr>
    <tr>
      <td></td>
      <td>CUBO_CANCER_DIGESTIVO</td>
      <td>Provincia, Nombre Provincia, Nombre Region, Region, Macrorregion, Año, Sexo, RangoEdad10, RangoEdad4080, MedianaRangoEdad10, MedianaRangoEdad4080, Poblacion, PoblacionRural, Fonasa A, Fonasa B, Fonasa C, Fonasa D, Fonasa, Isapre, Categoria, Cancer, RPC, Casos, Defunciones, Egresos_Defunciones, Egresos</td>
      <td>24948</td>
      <td>4.9 MB</td>
      <td>692 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>CUBO_CENTROS</td>
      <td>Codigo, Nombre, Nivel, Tipo, Complejidad, Comuna, Servicio, Latitud, Longitud, Nombre Comuna, Nombre Region, Porc_rural_comuna, Destino GES</td>
      <td>4582</td>
      <td>465.5 KB</td>
      <td>218 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>CUBO_DERIVACIONES_MDD</td>
      <td>Origen, Destino, Código prestación, COMGES, GES, Derivaciones</td>
      <td>6762</td>
      <td>317.1 KB</td>
      <td>33 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>CUBO_INSCRITOS_DERIVACIONES</td>
      <td>Origen, Sexo, Rango Edad, Año, Inscritos, Código prestación, Prestación, Tipo prestación, Especialidad, GES, Derivaciones</td>
      <td>906782</td>
      <td>76.1 MB</td>
      <td>576 KB</td>
    </tr>
    <tr>
      <td></td>
      <td>CUBO_COBERTURA_SERVICIOS</td>
      <td>idServicio, Codigo, RangoEdad4080, Año, Casos, Poblacion, PoblacionRural, IncidenciaEst, Consultas, Endoscopias, Confirmacion_C16, Confirmacion_HP, Tratamiento_C16, Tratamiento_HP, Egresos, Defunciones, Isapre, Fonasa A, Fonasa B, Fonasa C, Fonasa D, Servicio, Macrorregion, Sexo, Periodo</td>
      <td>10424</td>
      <td>2.4 MB</td>
      <td>429 KB</td>
    </tr>
  </tbody>
</table>
:::

The developed data lakehouse has demonstrated significant utility in supporting multiple epidemiological analyses and the dissemination of PhD thesis results.

### GC incidence scripts
The GC incidence scripts were used to prepare a congress poster and a scientific journal manuscript [@lagos_como_2025]. This last version was published in a [Github repository](https://github.com/rlagosb/GastricCancerIncidence) as part of a manuscript publication.

:::{figure} figure4.jpg
:label: fig:incidence
Variation of gastric cancer incidence across risk clusters and periods.

We identified a high-risk cluster in southern Chile and submitted our findings to scientific congresses and a peer-reviewed journal, sharing all data and scripts to ensure full reproducibility.
:::

### Diagnosis network scripts
The digestive cancer diagnosis network scripts were used to generate a report for the Ministry of Health,  to present a poster at a congress, and to prepare a scientific journal manuscript [@lagos_caracterizacion_2025]. A [GitHub repository](https://github.com/rlagosb/DigestiveCancerDiagnosisNetworks) was published to support the paper results.

:::{figure} figure5.png
:label: fig:networks
Gastric cancer diagnosis networks identified in rural and urban regions using NetworkX.

Networks identified in Maule Health Authority (left) and South East Metropolitan Health Authority (right). In total, eleven health authorities were divided into local diagnosis networks, allowing higher granularity of cancer diagnosis coverage metrics.
:::

### Diagnosis network scripts
Finally, the diagnosis coverage scripts were used for a congress poster and a manuscript [@lagos_effective_2026]. A [Github repository](https://github.com/rlagosb/GastricCancerEffectiveCoverage) was published to support the manuscript submission. The dashboard was published in streamlit cloud and used in congress presentations, but most oftenly for personal use: [tesisrenelagos.streamlit.app](https://tesisrenelagos.streamlit.app/).

## Discussion

We implemented a Data Lakehouse architecture to bridge the gap between the flexible, cost-effective storage of a data lake and the rigorous analytical structure of a data warehouse. We utilized a modular ELT workflow, loading raw datasets into our central repository with basic normalization before transforming them into multidimensional structures optimized for epidemiological analysis. Our pipeline organized both data and code into clear stages—progressing from Original Data to a finalized Data Warehouse—effectively merging the high-volume storage capacity of a lake with the relational consistency of a warehouse.

To manage the complexity of heterogeneous records, we adopted a D-ETL approach and implemented a "one dataset, one notebook" model. This provided the transparency of Python often lacking in GUI-based tools, where underlying transformation logic is frequently hidden. By embedding rules, assumptions, and mapping specifications as text cells directly alongside executable code, we created a single, human-readable file for each target dataset. This integrated format facilitated an iterative process of validation and debugging, ensuring that every step of the analytical logic remained visible and easily scrutinized. Furthermore, when using adequate keys and descriptions, narrows the gap between non data experts (like clinical professionals) so they not only can audite but also, re use it for other purposes.

By using Google Colab and Drive, we established a portable, cloud-native infrastructure that bypassed the lack of high-performance server capability often found in public health research. This ready-to-use environment turned standard cloud tools into a secure environment, ensuring that the analysis of data complied with institutional security policies while providing a shared, collaborative platform for researchers. By partitioning the pipeline into 38 discrete Jupyter Notebooks we avoided hitting the 12GB RAM and 50GB storage limits of the free-tier environment while providing an iterative, lightweight process that effectively managed 21 years of records for 20 million inhabitants.

This cloud-native environment facilitated statistical workflow through the `rpy2` library; we leveraged Python’s high-performance ecosystem for data engineering, while transitioning seamlessly into R for specialized modelling (e.g. Quasi-Poisson incidence). This interoperability eliminated the need for manual, error-prone data exports between software environments.

A critical choice in this implementation was the use of the Parquet columnar storage format, which provided substantial storage efficiency and performance gains. For example, a population dataset with nearly two million entries was compressed to just 3.2 MB on disk. Beyond speed, Parquet’s ability to store text metadata solved persistent encoding issues with Spanish characters in Chilean records, thereby safeguarding data integrity across heterogeneous sources.

Because the system follows the Effective Coverage framework, it can be easily replicated for colorectal or prostate cancer by simply mapping the "Need" and "Production" required datasets. Relational integrity is maintained through the data model based on the Chilean Norm for Health Information, ensuring that standardized attributes—such as sex, age, and county IDs—remain consistent across heterogeneous sources. This design allows for expansion by adding new records to existing tables with minimal structural changes. Furthermore, the strategic implementation of common Data Warehouse dimensions, such as DIM_Age, enables the integration of new datasets with varying granularities without requiring a rewrite of the core transformation logic.

### Limitations
While the free version of Google Colab facilitates access, it remains necessary to conduct a rigorous pre-loading anonymization process to ensure no Protected Health Information enters the public cloud environment. A standard implementation for non-anonymized records would require an ETLT (Extract-Transform-Load-Transform) workflow that enables a pre-load transformation stage to mask or encrypt sensitive identifiers before data reaches a cloud-based staging area, ensuring compliance with privacy regulations.

Furthermore, although the Chilean Norm for Health Information provided a robust national framework, it faces compatibility challenges compared to international benchmarks. Transitioning to the OMOP Common Data Model would be the next step for international and regional interoperability. Finally, while manual execution of D-ETL provided transparency and flexibility for academic research and PhD-scale projects, enterprise-level scaling would require automated pipelines and workflow orchestrators capable of managing higher data volume and velocity.

This study proves that high-resolution epidemiological research is feasible on standard hardware with nearly zero software costs. The Data Lakehouse, D-ELT, architecture serves as a scalable blueprint for researchers in resource-limited settings to conduct national-level research, lowering technical barriers while ensuring the reproducibility essential for public health research.
