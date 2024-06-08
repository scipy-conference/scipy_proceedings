---
# Ensure that this title is the same as the one in `myst.yml`
title: Algorithms to Determine Asteroid’s Physical Properties using Sparse and Dense Photometry, Robotic Telescopes and Open Data
abstract: |
  The pace of discovery of near-Earth asteroids outpaces current abilities to analyze them. Knowledge of an asteroid's physical properties is essential to deflect them. While dense photometry may yield continuous time-series measurements, which are useful for determining the rotation period of an asteroid, it is limited to a singular phase angle. On the other hand, sparse photometry provides non-continuous time-series measurements but for multiple phase angles, which are useful for determining absolute magnitude and size. I developed open-source algorithms that combine dense photometry from citizen scientists with sparse photometry from space and ground-based all-sky surveys using open data and mathematics to determine asteroids' albedo, size, rotation, strength, and composition. I took observations of the Didymos binary asteroid and combined them with GAIA, Zwicky Transient Facility, and ATLAS photometric sky surveys. My algorithm determined it to be 840m wide, with a 0.14 albedo, an 18.14 absolute magnitude, a 2.26-hour rotation period, rubble-pile strength, and an S-type composition. Didymos was the target of the 2022 NASA Double Asteroid Redirection Test (DART) mission. My algorithm successfully measured a 35-minute decrease in the mutual orbital period after the impact by the DART Mission. External sources validated the findings. Every citizen scientist is now a planetary defender.
---

## Introduction

There are over 1.3 million known asteroids, with hundreds of new near-Earth and main-belt asteroids discovered every month due to advanced detection techniques. However, the need for observations spanning multiple years, limited availability of telescopes, and narrow observation windows hinder detailed characterization of asteroids. To date, phase curves—which are essential for asteroid characterization—have been generated for only a few thousand of these asteroids. This slow pace of analysis hinders our planetary defense capabilities for deflecting potentially hazardous asteroids and limits our understanding of the solar system's evolution. This paper presents an innovative PhAst methodology developed using python algorithms for combining dense photometry from citizen scientists with sparse photometry from space and ground-based all-sky surveys to determine key physical characteristics of asteroids.

### Dense and Sparse Photometry
Dense photometry provides continuous time-series measurements, crucial for determining rotation periods, while sparse photometry offers non-continuous measurements across multiple phase angles, essential for absolute magnitude and size determination. Integrating both methods can overcome their individual limitations.

### Goals
Develop Python algorithms to integrate serendipitous asteroid observations with citizen-contributed and open datasets.
Apply these algorithms to planetary defense tests, such as NASA's DART mission.
Characterize large populations of asteroids to infer solar system compositional diversity.

## Methodology
### Development of Novel PhAst
PhAst, an open-source Python algorithm, integrates several years of sparse photometry from serendipitous asteroid observations with dense photometry from professional and citizen scientists. It generates phase curves whose linear components yield the asteroid’s geometric albedo and composition, while the non-linear brightness surge at small angles determines the absolute magnitude. This methodology allows for the creation of folded light curves to measure the asteroid’s rotation period and, for binary asteroids, their mutual orbital period.

### Data Sources and Integration
1. **Primary Asteroid Observations Using Robotic Telescopes:** Observation proposals were submitted to Alnitak Observatory, American Association of Variable Star Observers, Burke Gaffney Observatory, and Faulkes Telescope.
2. **Citizen Scientist Observations:** Observations submitted by backyard astronomers from locations such as Chile and the USA.
3. **Serendipitous Asteroid Observations in Sky Surveys:** Data from European Space Agency Gaia Data Release 3 and Zwicky Transient Facility (ZTF) Survey.
4. **Secondary Asteroid Databases:** Data from the Asteroid Lightcurve Database (ALCDEF) and Asteroid Photometric Data Catalog (PDS) 3rd update.

For searching asteroids in the ZTF dataset, the FINKS portal (https://fink-portal.org/about, https://fink-broker.org/about) was utilized, which allowed searching asteroids by their Minor Planet Center (MPC) number. Similarly, asteroids in the GAIA dataset were searched using the Solar System Objects database of Gaia DR3 (https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu4sso/sec_cu4sso_intro/)

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
The Didymos binary asteroid, targeted by NASA's 2022 Double Asteroid Redirection Test (DART) mission, was selected for a detailed case study. Initial observations determined Didymos to be 840 meters wide, with a 0.14 albedo, an 18.14 absolute magnitude, a 2.26-hour rotation period, rubble-pile strength, and an S-type composition. These properties were derived by applying the PhAst algorithm to a combination of dense and sparse photometric data.

### Impact Analysis
PhAst successfully measured a 35-minute decrease in the mutual orbital period following the DART mission's impact. External sources validated these findings, demonstrating the algorithm's accuracy and reliability. The change in the mutual orbital period provided critical data on the effectiveness of the DART mission in altering the asteroid's trajectory, a key goal of planetary defense strategies.

## Results
PhAst was used to generate phase curves for over 2100 asteroids in 100 hours on a home computer. 
Determining Physical Properties of Target Asteroids of Space Missions and Understudied Asteroids
PhAst was used to generate phase curves and determine the physical properties of various target asteroids of space missions and understudied asteroids. The results include:

### Targets of NASA LUCY Mission:
3548 Eurybates: H=9.75 ± 0.05, G=0.11, Albedo = 0.05
10253 Westerwald: H=15.33 ± 0.05, G=0.17, Albedo = 0.21

### Targets of UAE Mission:
269 Justitia: H=9.93 ± 0.09, G=0.11, Albedo = 0.09
15094 Polymele: H=11.69 ± 0.07, G=0.18, Albedo = 0.05

### Binary Asteroids:
3378 Susanvictoria: H=13.83 ± 0.05, G=0.27, Albedo = 0.19
2825 Crosby: H=13.33 ± 0.06, G=0.11, Albedo = 0.07

### Understudied Asteroids:
2006 MG13: H=15.94 ± 0.08, G=0.21, Albedo = 0.19
2007 AD11: H=15.76 ± 0.11, G=0.13, Albedo = 0.13

The physical properties of the binary asteroids were submitted to the binary asteroid working group. This rapid analysis capability highlights PhAst's potential for large-scale asteroid characterization, enabling detailed studies of large populations of asteroids in a relatively short time.

## Discussions
### Determining the Success of Asteroid Deflection
The success of the DART mission was evaluated by analyzing the change in the orbital path of Dimorphos, the moonlet of Didymos, after deflection. Applying Kepler's Third Law, the pre-impact orbital period of 11.91 hours and post-impact orbital period of 11.34 hours were used to calculate an orbital radius change of 0.04 km. This change confirms the effectiveness of the DART mission in altering the asteroid's trajectory, a crucial component of planetary defense.

### Determining Asteroid Strength
Asteroid strength can be inferred from the rotation period. If the rotation period is less than 2.2 hours, the asteroid must be a strength-bound single rock; otherwise, it would fly apart. If the rotation period is greater than 2.2 hours and the diameter is more than 150 meters, the asteroid is likely a rubble pile held together by mutual gravitation. This information is vital for assessing the structural integrity of asteroids and planning deflection missions.

### Determining Asteroid Taxonomy
Asteroid taxonomy (chemical composition) can be determined from geometric albedo. C-type asteroids have lower albedo, S-type and M-type asteroids have moderate albedo, and rare E-type asteroids have the highest albedo. The taxonomical distribution provides insights into the conditions of the early solar system based on the spatial distribution of asteroid types. Understanding these compositions helps in determining the origins and evolutionary history of these asteroids.

### Early Solar System Conditions
The taxonomical distributions of carbonaceous, siliceous, and metallic asteroids in the main belt were compiled. Over 58% of the asteroids characterized by PhAst are carbonaceous, showing they are the most abundant type in our Solar System. Their abundance increases with distance from the Sun, reaching nearly 75% in the outer region of the main belt compared to over 45% in the inner region. The spatial distribution of asteroid types provides clues about the conditions in the early solar system. Carbonaceous asteroids are more abundant in the outer regions of the main belt, suggesting they formed in cooler, volatile-rich environments. In contrast, siliceous and metallic asteroids are more common in the inner regions, indicating formation in hotter, more metal-rich conditions. These patterns help reconstruct the processes that shaped our solar system's formation and evolution.

### Errors and Limitations
Photometry was performed on images with a Signal-to-Noise Ratio (SNR) > 100, yielding a measurement uncertainty of 0.01. The average error in phase curve fitting was 0.10. Limited processing power restricted the preciseness of the best fit for rotation and mutual orbital periods to two significant digits. These limitations highlight the need for more powerful computational resources and more precise observational data to improve the accuracy of asteroid characterization.

## Conclusions
PhAst represents a significant advancement in asteroid characterization, combining dense and sparse photometry to yield comprehensive insights into asteroid properties. The successful application of PhAst to the Didymos binary asteroid and over 2100 other asteroids demonstrates its potential for large-scale use. By engaging citizen scientists, we can accelerate asteroid analysis and enhance our planetary defense strategies.

## Future Work
PhAst will serve as a powerful tool for accelerating the analysis of data produced by the Legacy Survey of Space and Time (LSST), set to begin in 2025. Over a decade, LSST aims to observe over 5 million asteroids across various filters, generating a nightly data volume of 20TB. PhAst can also be expanded to exoplanetary atmosphere characterization by adapting its methodology to analyze the light curves from transiting exoplanets in multiple filters. This expansion would open new avenues for research in planetary science and beyond.

## Project Impact
The PhAst algorithm has been made open-source, and training modules have been developed for citizen scientists. These modules, created using Jupyter Notebooks, are designed for use by high school students and citizen scientists to support their engagement in asteroid characterization and planetary defense. Training on using open data for asteroid categorization has been provided to over 1,500 students during "Space Day" and "Asteroid Day" events in collaboration with observatories and community organizations.

## Acknowledgments
The development and application of PhAst have been possible thanks to contributions from numerous observatories, citizen scientists, and research institutions. Special thanks to the teams behind GAIA, Zwicky Transient Facility (ZTF), ATLAS, and other photometric surveys for providing the data that made this research possible. We also acknowledge the support of various citizen science communities and educational organizations for their collaboration and participation.

## References
Shevchenko, V. G., et al. "Geometric Albedo Range values."
NASA/JPL. "DART Mission Overview."
Zwicky Transient Facility (ZTF). "Photometric Data Release."
GAIA Star Catalog. "Data Release 3."
Asteroid Lightcurve Database (ALCDEF). "Observation Submissions."
Asteroid Photometric Data Catalog (PDS). "Third Update."
Fink Broker “ZTF Minor Planet Photometric Data Release.”
