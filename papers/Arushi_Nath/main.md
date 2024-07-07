---
# Ensure that this title is the same as the one in `myst.yml`
title: Algorithms to Determine Asteroid’s Physical Properties using Sparse and Dense Photometry, Robotic Telescopes and Open Data
abstract: |
  The rapid pace of discovering asteroids due to advancements in detection techniques outpaces current abilities to analyze them comprehensively. Understanding an asteroid's physical properties is crucial for effective deflection strategies and improves our understanding of the solar system's formation and evolution. Dense photometry provides continuous time-series measurements valuable for determining an asteroid's rotation period, yet is limited to a singular phase angle. Conversely, sparse photometry offers non-continuous measurements across multiple phase angles, essential for determining an asteroid's absolute magnitude, albedo (reflectivity), and size. This paper presents open-source algorithms that integrate dense photometry from citizen scientists with sparse photometry from space and ground-based all-sky surveys to determine asteroids' albedo, size, rotation, strength, and composition. 
Applying the algorithms to the Didymos binary asteroid, combined with data from GAIA, the Zwicky Transient Facility, and ATLAS photometric sky surveys, revealed Didymos to be 840 meters wide, with a 0.14 albedo, an 18.14 absolute magnitude, a 2.26-hour rotation period, rubble-pile strength, and an S-type composition. Didymos was the target of the 2022 NASA Double Asteroid Redirection Test (DART) mission. The algorithm successfully measured a 35-minute decrease in the mutual orbital period following the DART mission, equating to a 40-meter reduction in the mutual orbital radius, proving a successful deflection. Analysis of the broader asteroid population highlighted significant compositional diversity, with a predominance of carbonaceous (C-type) asteroids in the outer regions of the asteroid belt and siliceous (S-type) and metallic (M-type) asteroids more common in the inner regions. These findings provide insights into the diversity and distribution of asteroid compositions, reflecting the conditions and processes of the early solar system. 
This work empowers citizen scientists to become planetary defenders, contributing significantly to planetary defense and enhancing our understanding of solar system composition and evolution.

---

## Introduction

### Background
There are over 1.3 million known asteroids, and advanced detection techniques lead to the discovery of hundreds of new near-Earth and main-belt asteroids every month. Studying these asteroids provides valuable insights into the early solar system's formation and evolution. Phase curves, which illustrate the change in an asteroid's brightness as its phase angle (the angle between the observer, asteroid, and Sun) changes, are essential for asteroid characterization. Understanding near-Earth asteroids is crucial because it allows for the development of effective deflection strategies, which are vital for preventing potential collisions with Earth and safeguarding our planet from catastrophic impacts.

### Research Problem
Despite advancements in detection techniques, the need for observations spanning multiple years, limited telescope availability, and narrow observation windows hinder detailed characterization of asteroids. To date, phase curves have been generated for only a few thousand asteroids. This slow pace of analysis hinders our planetary defense capabilities for deflecting potentially hazardous asteroids and limits our understanding of the solar system's evolution.

### Related Work
Recent efforts in the field have focused on various approaches to combine dense and sparse photometric datasets. For instance, the Pan-STARRS survey has used sparse photometry to estimate the absolute magnitudes and rotation periods of asteroids, while the Asteroid Terrestrial-impact Last Alert System (ATLAS) provides sparse photometry data for many asteroids observed across different phase angles. Studies by Shevchenko et al. (2019) have explored methods to derive phase integrals and geometric albedos from sparse data. On the dense photometry side, projects like the Zwicky Transient Facility (ZTF) and Gaia Data Release 3 (DR3) have contributed extensive datasets valuable for continuous observations of asteroid brightness variations. However, these efforts often face challenges in data integration due to differing observational cadences, filters, and coverage.

### Research Objectives
This paper presents an innovative methodology, PhAst, developed using Python algorithms to combine dense photometry from citizen scientists with sparse photometry from space and ground-based all-sky surveys to determine key physical characteristics of asteroids. The specific objectives of this research are to:
1. Develop Python algorithms to integrate serendipitous asteroid observations with citizen-contributed and open datasets.
2. Apply these algorithms to planetary defense tests, such as NASA's DART mission.
3. Characterize large populations of asteroids to infer compositional diversity in our solar system.

### Significance of the Study
This study offers significant contributions to the field of asteroid characterization and planetary defense. By integrating dense and sparse photometry, the PhAst algorithm provides a comprehensive method for determining the physical properties of asteroids. The open-source nature of the algorithm encourages collaboration and improvements from a global community of researchers and citizen scientists, enhancing its robustness and accelerating advancements in the field. Furthermore, the study empowers citizen scientists to actively participate in planetary defense, contributing valuable data and insights that enhance our understanding and preparedness for potential asteroid impacts.

### Application in Planetary Defense: NASA’s DART Mission
The NASA Double Asteroid Redirection Test (DART) mission was designed to test and validate methods to protect Earth from hazardous asteroid impacts by demonstrating the kinetic impactor technique. It involved sending a spacecraft to collide with an asteroid to change its trajectory. PhAst provides a detailed pre- and post-impact analysis of the target asteroid, Didymos, and its moonlet, Dimorphos.

## Methodology
### Overview

The PhAst algorithm integrates dense and sparse photometry data to determine the physical properties of asteroids. Dense photometry provides continuous time-series measurements, crucial for determining rotation periods, while sparse photometry offers non-continuous measurements across multiple phase angles, essential for absolute magnitude and size determination. Integrating both methods can overcome their individual limitations.

### Development of Novel Open-Source PhAst
PhAst integrates several years of sparse photometry from serendipitous asteroid observations with dense photometry from professional and citizen scientists. **See Figure 1.** The algorithm effectively combines continuous light data (dense photometry) and infrequent light data (sparse photometry) by creating phase curves whose linear components yield the asteroid’s geometric albedo and composition, while the non-linear brightness surge at small angles determines the absolute magnitude. This methodology allows for the creation of folded light curves to measure the asteroid’s rotation period and, for binary asteroids, their mutual orbital period. Being open-source, the PhAst algorithm allows for collaboration and improvements from a global community of researchers and citizen scientists, enhancing its robustness and accelerating advancements in asteroid characterization.


```{figure} figure1.png
:name: fig1
:align: center

Flowchart Showing Data Integration Process of PhAst
```

### Data Sources and Integration
1. **Primary Asteroid Observations Using Robotic Telescopes:** Observation proposals were submitted to Alnitak Observatory, American Association of Variable Star Observers, Burke Gaffney Observatory, and Faulkes Telescope.
2. **Citizen Scientist Observations:** Observations submitted by backyard astronomers from locations such as Chile and the USA.
3. **Serendipitous Asteroid Observations in Sky Surveys:** Data from European Space Agency Gaia Data Release 3 and Zwicky Transient Facility (ZTF) Survey.
4. **Secondary Asteroid Databases:** Data from the Asteroid Lightcurve Database (ALCDEF) and Asteroid Photometric Data Catalog (PDS) 3rd update.

For searching asteroids in the ZTF dataset, the FINKS portal was utilized, which allowed searching asteroids by their Minor Planet Center (MPC) number. Similarly, asteroids in the GAIA dataset were searched using the Solar System Objects database of Gaia DR3.

### Observational Process
1. **Identify Known Stars and Asteroids:** Using the GAIA Star Catalog and HORIZONS Asteroid Catalog, known stars and asteroids are identified and centroided in images. This step ensures that the exact positions of celestial objects are accurately determined, which is crucial for subsequent analysis.
2. **Determine Optimal Aperture Size:** Differential photometry is used to calculate the asteroid's instrumental magnitude by determining the optimal aperture size that balances brightness measurement and noise. Too small an aperture may not capture the full brightness of the asteroid, while too large an aperture may include excessive background noise.
3. **Select Suitable Comparison Stars:** Comparison stars with stable brightness are selected to remove the effects of seeing conditions and determine the asteroid's computed magnitude. This step is important to ensure that variations in observed brightness are due to the asteroid itself and not due to atmospheric conditions or instrumental errors.
4. **Remove Distance Effects:** Effects of the asteroid's changing distance from the Sun and the observer are removed to find the reduced magnitude. This normalization allows for a more accurate comparison of observations taken at different times and distances.
5. **Generate Phase Curves:** Phase angle effects are removed to generate phase curves, which help find the absolute magnitude and linear slope. These phase curves provide insights into the reflectivity and surface properties of the asteroid.
6. **Determine Rotation and Orbital Periods:** Composite light curves are used to find the asteroid's rotation period and, for binary asteroids, the mutual orbital period. This analysis reveals the dynamic characteristics of the asteroid, including its spin state and orbital interactions with companion bodies.

### Python Tools and Libraries
The development and implementation of PhAst heavily relied on various Python tools and libraries:
- **Python:** The primary programming language used for developing PhAst.
- **NumPy:** Used for numerical computations and handling large datasets efficiently.
- **Matplotlib:** Utilized for plotting phase curves and light curves, visualizing the data, and generating graphs for analysis.
- **AstroPy:** Employed for astronomical calculations and handling astronomical data, such as coordinate transformations and time conversions.

## Case Study: Didymos Binary Asteroid
### Initial Observations
The Didymos binary asteroid, targeted by NASA's 2022 Double Asteroid Redirection Test (DART) mission, was selected for a detailed case study. Initial observations determined Didymos to be 840 meters wide, with a 0.14 albedo, an 18.14 absolute magnitude (a measure of its intrinsic brightness), a 2.26-hour rotation period, rubble-pile strength (indicating it is a loose collection of rocks held together by gravity), and an S-type composition (indicating it is made of stony or siliceous minerals). These properties were derived by applying the PhAst algorithm to a combination of dense and sparse photometric data.

### Impact Analysis
PhAst successfully measured a 35-minute decrease in the mutual orbital period following the DART mission's impact. External sources validated these findings, demonstrating the algorithm's accuracy and reliability. The change in the mutual orbital period provided critical data on the effectiveness of the DART mission in altering the asteroid's trajectory, a key goal of planetary defense strategies.

## Results
PhAst was used to generate phase curves for over 2100 asteroids in 100 hours on a home computer, including data-retrieval time. The physical properties of various target asteroids of space missions and understudied asteroids were determined, including targets of the NASA LUCY Mission, UAE Mission, binary asteroids, and understudied asteroids. **See figure 2.** The rapid analysis capability highlights PhAst's potential for large-scale asteroid characterization, enabling detailed studies of large populations of asteroids in a relatively short time.

```{figure} figure3.png
:name: fig2
:align: center

Physical Properties of Target Asteroids of Space Missions and Understudied Asteroids Determined 
```
### Determining Physical Properties of Target Asteroids of Space Missions and Understudied Asteroids
PhAst was used to generate phase curves and determine the physical properties of various target asteroids of space missions and understudied asteroids. The results include:

#### NASA LUCY Mission Targets
The NASA LUCY mission aims to explore Trojan asteroids, which share Jupiter's orbit around the Sun. Understanding these asteroids can provide insights into the early solar system since Trojans are considered remnants of the primordial material that formed the outer planets.

- **3548 Eurybates:**
  - Absolute Magnitude (H) = 9.75 ± 0.05
  - Slope Parameter (G) = 0.11
  - Albedo = 0.05
  * Relevance: Eurybates is the largest and presumably the most ancient member of the Eurybates family, offering a window into the conditions of the early solar system.

- **10253 Westerwald:**
  - Absolute Magnitude (H) = 15.33 ± 0.05
  - Slope Parameter (G) = 0.17
  - Albedo = 0.21
  * Relevance: Westerwald's high albedo suggests it might be a fragment from a larger parent body, providing clues about collisional processes in the early solar system.


#### UAE Mission Targets
The UAE space mission to explore asteroids aims to study their composition, structure, and history, contributing to our understanding of asteroid formation and the evolution of the solar system.

- **269 Justitia:**
  - Absolute Magnitude (H) = 9.93 ± 0.09
  - Slope Parameter (G) = 0.11
  - Albedo = 0.09
  * Relevance: Justitia's relatively low albedo indicates a carbonaceous composition, which can help researchers understand the distribution of organic materials in the solar system.

- **15094 Polymele:**
  - Absolute Magnitude (H) = 11.69 ± 0.07
  - Slope Parameter (G) = 0.18
  - Albedo = 0.05
  * Relevance: Polymele's properties suggest it is a primitive body, providing valuable information about the early solar system's building blocks.


#### Binary Asteroids
Understanding binary asteroids, where two asteroids orbit each other, can offer insights into the formation and evolutionary history of these systems. The mutual orbital period and other physical properties provide data on their dynamics and interactions.

- **3378 Susanvictoria:**
  - Absolute Magnitude (H) = 13.83 ± 0.05
  - Slope Parameter (G) = 0.27
  - Albedo = 0.19
  * Relevance: Studying binary systems like Susanvictoria helps in understanding the processes that lead to the formation of binary asteroids and their subsequent evolution.

- **2825 Crosby:**
  - Absolute Magnitude (H) = 13.33 ± 0.06
  - Slope Parameter (G) = 0.11
  - Albedo = 0.07
  * Relevance: Crosby's characteristics can provide insights into the collisional history and mechanical properties of binary asteroid systems.
The physical properties of the binary asteroids were submitted to the binary asteroid working group.

#### Understudied Asteroids
Characterizing understudied asteroids expands our knowledge of the diversity and distribution of asteroid properties in the solar system.

- **2006 MG13:**
  - Absolute Magnitude (H) = 15.94 ± 0.08
  - Slope Parameter (G) = 0.21
  - Albedo = 0.19
  * Relevance: Detailed study of asteroids like 2006 MG13 helps fill gaps in our understanding of the physical and compositional diversity of asteroids.

- **2007 AD11:**
  - Absolute Magnitude (H) = 15.76 ± 0.11
  - Slope Parameter (G) = 0.13
  - Albedo = 0.13
  * Relevance: Investigating such understudied bodies contributes to a more complete picture of asteroid population characteristics and their evolutionary paths.


## Discussions
### Determining the Success of Asteroid Deflection
The success of the DART mission was evaluated by analyzing the change in the orbital path of Dimorphos, the moonlet of Didymos, after deflection. Applying Kepler's Third Law, the pre-impact orbital period of 11.91 hours and post-impact orbital period of 11.34 hours were used to calculate an orbital radius change of 0.04 km. This change confirms the effectiveness of the DART mission in altering the asteroid's trajectory, a crucial component of planetary defense.

### Determining Asteroid Strength
Asteroid strength can be inferred from the rotation period. This inference is based on the fact that an asteroid's structural integrity must be sufficient to withstand the centrifugal forces generated by its rotation. If the rotation period is less than 2.2 hours, the asteroid must be a strength-bound single rock; otherwise, it would fly apart due to centrifugal forces exceeding the gravitational binding forces. This criterion is supported by studies such as those by Pravec and Harris (2000), who observed that most asteroids with rotation periods shorter than 2.2 hours are smaller than 150 meters and are likely monolithic. For larger asteroids, the rubble-pile structure is held together by self-gravity rather than cohesive forces, making them prone to disaggregation at faster rotation rates. This information is vital for assessing the structural integrity of asteroids and planning deflection missions.

### Determining Asteroid Taxonomy
Asteroid taxonomy (chemical composition) can be determined from geometric albedo. C-type asteroids have lower albedo, S-type and M-type asteroids have moderate albedo, and rare E-type asteroids have the highest albedo. (S-type asteroids are made of stony or siliceous minerals, while C-type and M-type refer to carbonaceous and metallic compositions, respectively.) The taxonomic distribution provides insights into the conditions of the early solar system based on the spatial distribution of asteroid types. Understanding these compositions helps in determining the origins and evolutionary history of these asteroids.

### Early Solar System Conditions
The taxonomical distributions of carbonaceous, siliceous, and metallic asteroids in the main belt were compiled. Over 58% of the asteroids characterized by PhAst are carbonaceous, showing they are the most abundant type in our Solar System. Their abundance increases with distance from the Sun, reaching nearly 75% in the outer region of the main belt compared to over 45% in the inner region. **See figure 3.** This finding is consistent with previous research in the field, such as studies by DeMeo and Carry (2014), which indicate that carbonaceous asteroids are prevalent in the outer asteroid belt.
Characterizing asteroid populations helps us better understand the diversity of compositions in the solar system by providing a detailed inventory of the different types of asteroids and their distribution. This information is crucial for several reasons:
- **Formation Conditions:** Different types of asteroids formed under varying conditions in the early solar system. For example, carbonaceous (C-type) asteroids, which are rich in organic compounds, are more prevalent in the outer regions of the asteroid belt, suggesting formation in cooler, volatile-rich environments. In contrast, siliceous (S-type) and metallic (M-type) asteroids are more common in the inner regions, indicating formation in hotter, more metal-rich conditions.
- **Evolutionary Processes:** By studying the physical and chemical properties of asteroids, we can infer the processes that have shaped their evolution. This includes understanding how collisions, thermal processes, and space weathering have affected their surfaces and internal structures.

```{figure} figure4.png
:name: fig3
:align: center

Spatial Distribution of Asteroid Types
```

### Errors and Limitations
Photometry was performed on images with a Signal-to-Noise Ratio (SNR) > 100, yielding a measurement uncertainty of 0.01. The average error in phase curve fitting was 0.10. Limited processing power restricted the preciseness of the best fit for rotation and mutual orbital periods to two significant digits. These limitations highlight the need for more powerful computational resources and more precise observational data to improve the accuracy of asteroid characterization.

## Conclusions
PhAst represents a significant advancement in asteroid characterization, combining dense and sparse photometry to yield comprehensive insights into asteroid properties. The successful application of PhAst to the Didymos binary asteroid and over 2100 other asteroids demonstrates its potential for large-scale use. By engaging citizen scientists, we can accelerate asteroid analysis and enhance our planetary defense strategies.

## Future Work
PhAst will serve as a powerful tool for accelerating the analysis of data produced by the Legacy Survey of Space and Time (LSST), set to begin in 2025. Over a decade, LSST aims to observe over 5 million asteroids across various filters, generating a nightly data volume of 20TB. The specific benefits and new opportunities that PhAst's applications might bring include:
- **Enhanced Planetary Defense:** By rapidly characterizing large populations of asteroids, including potentially hazardous asteroids (PHAs), PhAst can provide detailed analysis that are crucial for developing effective deflection strategies, thereby enhancing planetary defense capabilities.
- **Comprehensive Asteroid Mapping:** The integration of dense and sparse photometry allows for the creation of more accurate and comprehensive maps of asteroid distributions and compositions in the solar system. This can provide valuable insights into the formation and evolution of the solar system, aiding both scientific research and educational initiatives.
- **Resource Identification and Utilization:** PhAst's ability to determine the physical and compositional properties of asteroids can aid in identifying asteroids rich in valuable minerals or water. This opens up new opportunities for asteroid mining and resource utilization, which could support long-term space exploration and the development of space infrastructure.
- **Support for Future Space Missions:** PhAst can be used to provide detailed pre and post mission characterization of target asteroids for upcoming space missions including NASA’s OSIRIS-APEX which will fly-by near-Earth asteroid Apophis on April 23, 2029, JAXA’s Hayabusa2 SHARP to explore two asteroids, 2001 CC21 and 1998 KY26, and China’s first kinetic impact deflection test mission would target the near-Earth asteroid 2015 XF261 with a launch in 2027. 
- **Exoplanetary Atmosphere Characterization:** PhAst can be expanded to exoplanetary atmosphere characterization by adapting its methodology to analyze the light curves from transiting exoplanets in multiple filters. This expansion would allow researchers to study the atmospheres of distant planets, providing insights into their composition, climate, and potential habitability.
- **Citizen Science and Public Engagement:** By making PhAst open-source and developing training modules for citizen scientists, the project promotes public engagement in scientific research. This democratization of science enables a wider community to contribute to and benefit from cutting-edge research, fostering a culture of curiosity and collaboration.

## Project Impact
The PhAst algorithm has been made open-source, and training modules have been developed for citizen scientists. These modules, created using Jupyter Notebooks, are designed for use by high school students and citizen scientists to support their engagement in asteroid characterization and planetary defense. Training on using open data for asteroid categorization has been provided to over 1,500 students during "Space Day" and "Asteroid Day" events in collaboration with observatories and community organizations such as Royal Astronomical Society of Canada. See link to Github: https://github.com/Spacegirl123/Asteroid-Characterization-By-PhAst

## Acknowledgments
The development and application of PhAst have been possible thanks to contributions from numerous observatories, citizen scientists, and research institutions. Special thanks to the teams behind GAIA, Zwicky Transient Facility (ZTF), ATLAS, and other photometric surveys for providing the data that made this research possible. I also acknowledge the support of various citizen science communities and educational organizations for their collaboration and participation.

## References

[1] Center for Near-Earth Object Studies. Total number of asteroids discovered monthly. Retrieved from https://cneos.jpl.nasa.gov/stats/totals.html

[2] NASA/Johns Hopkins University Applied Physics Laboratory. (2022, March). NASA's first planetary defense technology demonstration to collide with asteroid in 2022. https://www.nasa.gov/feature/nasa-s-first-planetary-defense-technology-demonstration-to-collide-with-asteroid-in-2022

[3] Shevchenko, V. G., et al. (2019). Phase integral of asteroids. Astronomy & Astrophysics, 626(A87). https://doi.org/10.1051/0004-6361/201935588

[4] Talbert, T. (2022, October 11). NASA DART imagery shows changed orbit of target asteroid. NASA. https://www.nasa.gov/solar-system/nasa-dart-imagery-shows-changed-orbit-of-target-asteroid/

[5] Jet Propulsion Laboratory. (n.d.). Small-Body Database Lookup. https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=65803

[6] Fink Broker. (n.d.). ZTF Minor Planet Photometric Data Release. https://fink-portal.org/

[7] European Space Agency. (n.d.). Gaia Data Release 3. https://www.cosmos.esa.int/web/gaia/dr3

[8] ALCDEF. (n.d.). Asteroid Lightcurve Photometry Database. https://alcdef.org/

[9] Planetary Science Institute. (n.d.). Asteroid Photometric Catalog (APC) "Third Update." https://sbn.psi.edu/pds/resource/apc.html

[10] Pravec, P., & Harris, A. W. (2000). Fast and Slow Rotation of Asteroids. Icarus, 148(1), 12-20. https://doi.org/10.1006/icar.2000.6482

[11] DeMeo, F. E., & Carry, B. (2014). Solar System evolution from compositional mapping of the asteroid belt. Nature, 505(7485), 629-634. https://doi.org/10.1038/nature12908
