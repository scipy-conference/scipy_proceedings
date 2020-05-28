:author: Mark Wickert
:email: mwickert@uccs.edu
:institution: University of Colorado Colorado Springs

:author: David Peckham
:email: dpeckham@uccs.edu
:institution: University of Colorado Colorado Springs

:video: http://www.youtube.com/watch?v=dhRUe-gz690

--------------------------------------------------------------------------------
Matched Filter Mismatch Losses in MPSK and MQAM Using Semi-Analytic BEP Modeling
--------------------------------------------------------------------------------

.. class:: abstract

   The focus of this paper is the bit error probability (BEP) performance 
   degradation when the transmit and receive pulse shaping filters are 
   mismatched. The modulation schemes considered are MPSK and MQAM. 
   In the additive white Gaussian noise (AWGN) channel both spectral 
   efficiency and noise mitigation is commonly achieved by using 
   square-root raised cosine (SRC) pulse shaping at both the transmitter 
   and receiver. The novelty of this paper primarily lies in the use 
   semi-analytic BEP simulation for conditional error probability calculations, with transmit and receive filter mismatch, the optional inclusion of a small FIR equalizer. For lower order MPSK and MQAM, i.e., 8PSK and 16QAM :math:`E_b/N_0` power degradation at :math:`\text{BEP} = 10^{-6}` is 0.1 dB when the excess bandwidth mismatch tx/rx = 0.25/0.35 or 0.35/0.25, but quickly grows as the modulation order increases and/or the mismatch increases. 


.. class:: keywords

   digital modulation, pulse shaping, phase-shift keying, 
   quadrature amplitude modulation 


Introduction
------------

In the early days of satellite and space communications, digital
modulation schemes focused on constant envelope waveforms, in particular
phase-shift keying (PSK), with rectangular pulse shaping [Lindsey]_. 
The need for spectral efficiency is ever present in modern communication 
systems [Ziemer]_, [Proakis]_, and [Goldsmith]_. The use of pulse 
shaping makes spectral efficiency possible, at the expense of non-constant 
envelope waveforms [Ziemer]_. Today m-ary PSK (MPSK) and
high density m-ary quadrature amplitude modulation (MQAM), both with
pulse shaping, are found in satellite communications.

In certain applications the precise pulse shape used by the transmitter
is not known by the receiver. The use of an equalizer is always an
option in this case, but it adds complexity and for burst mode systems,
e.g., TDMA, the convergence time of an adaptive equalizer is another
issue to deal with.

The focus of this paper is the bit error probability (BEP) performance
degradation when the transmit and receive pulse shaping filters are
mismatched. The modulation schemes considered are MPSK and MQAM. In the
additive white Gaussian noise (AWGN) channel both spectral efficiency
and noise mitigation is commonly achieved by using square-root raised
cosine (SRC) pulse shaping at both the transmitter and receiver. The
system block diagram is shown in Figure :ref:`Topblock`.
Notice that this block diagram also shows a symbol-spaced equalizer to
allow for the exploration of potential performance improvement, subject
to how much complexity can be afforded, and the need for rapid
adaptation in the case of burst mode transmission. We take advantage of
semi-analytic simulation techniques described in 
[Tranter]_ to allow fast and efficient performance
evaluation. All of the simulation software is written in open-source
Python.


.. figure:: Block_top.pdf
   :scale: 85%
   :align: center
   :figclass: htb

   System top level block diagram showing transmit and receive pulse
   shaping filters with mismatch, i.e., :math:`\alpha_t \neq \alpha_r`,
   and optionally the inclusion of an adaptive equalizer. :label:`Topblock`


Other authors such as, [Harris]_ and [Xing]_, have made mention of matched filter
mismatch, but not in the same context as we consider in this paper.
Harris is primarily driven by sidelobe reduction from the transmitter
perspective, while Xing wishes to avoid the added complexity of an
equalizer by using a specially designed receiving filter. Here we are
concerned with the situation where the receiver does not always know the
exact design of the transmit pulse shape filter, in particular the
excess bandwidth factor.

The remainder of this paper is organized as follows. We first consider
residual errors at the matched filter output when using a simple
truncated square-root raised cosine (SRC) finite impulse response (FIR).
In particular we consider filter lengths of :math:`\pm L` symbols in
duration, where :math:`L=6`, but also make this a design parameter. We
then briefly explain how a symbol-spaced adaptive equalizer can be
inserted at the output of the matched filter to compensate for pulse
shape mismatch. We then move on to briefly review the concept of
semi-analytic (SA)simulation and the develop conditional error
probability expressions for MPSK and MQAM. Finally we move into
performance.


Pulse Shaping Filter Considerations
-----------------------------------

The pulse shape used for this matched filter mismatch study is the
discrete-time version of the squareroot raised-cosine pulse shape:

.. math::
   :label: SRCpulse

   p_\text{SRC}(t) = \begin{cases}
           1 - \alpha +4\alpha/\pi, & t = 0 \\
           \frac{\alpha}{\sqrt{2}}\Big[\big(1+\frac{2}{\pi}\big)\sin\big(\frac{\pi}{4\alpha}\big) \\
           \quad\quad\big(1-\frac{2}{\pi}\big)\cos\big(\frac{\pi}{4\alpha}\big)\Big], & t = 
           \pm \frac{T}{4\alpha} \\
           \Big\{\sin\big[\pi t(1-\alpha)/T\big] + \\
           4\alpha t\cos\big[\pi t(1+\alpha)/T\big]/T\Big\}/ \\
           \Big\{\pi t\big[1 - (4\alpha t/T)^2\big]/T\Big\}^{-1}, & \text{otherwise}
       \end{cases}

where :math:`T` is the symbol period. In the literature this is often
referred to as the ideal root raised cosine filter (RRC)
[Rappaport]_. The name used here is square-root
raised cosine (SRC). For realizability considerations the discrete-time
transmit pulse shaping filter and receiver matched filter are obtained
by time shifting and truncating and then sampling by letting
:math:`t\rightarrow n T_s`.

Residual errors at the matched filter output when using a simple
truncated square-root raised cosine (SRC) finite impulse response (FIR)
are noted in both [Harris]_ and [Xing]_. Later in this paper we make the filter
length :math:`\pm 6` symbols in duration, but also make this a design
parameter as shown in Figure :ref:`SRCresidual`.

.. figure:: Residual_compare_4QAM.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Matched SRC filters at transmit and receiver showing residual error
   due to FIR filter truncation of the doubly infinite pulse response
   [Rappaport]_, for a nominal maximum eye opening
   of :math:`\pm 1`. :label:`SRCresidual`


Semi-Analytic Bit Error Probability
-----------------------------------

Semi-analytic BEP (SA-BEP) calculation allows for fast and efficient
analysis when a linear channel exists from the WGN noise injection point
to the receiver detector [Tranter]_. A block
diagram, which applies to the matched filter mismatch scenario of this
paper, is shown in Figure :ref:`BlockSAsim`. The variable
:math:`z_k` is the complex baseband detector decision statistic, as the
receiver matched filter is sampled at the symbol rate, :math:`R_s=1/T`,
nominally at the maximum eye opening. ISI is present in :math:`z_k` due
to pulse shape mismatch and other impairments such as timing error,
static phase error, and even phase jitter. This corresponds to an
ensemble of conditional Gaussian probabilities. The variance
:math:`\sigma_w^2`, for each the real/imaginary parts
(inphase/quadrature), is calculated using

.. math::
   :label: noisePwr

   \sigma_w^2 = N_0\cdot \sum_{n=0}^{N_\text{taps}-1} |p_r[n]|^2,

where the variance of the additive white Gaussian noise is denoted
:math:`N_0` and :math:`p_r[n]` is the matched filter impulse response
consisting of :math:`N_\text{taps}`. The value of :math:`\sigma_w` found
in the conditional error probability of the following subsections, is a
function of :math:`N_0` which is set to give the desired average
received energy per symbol :math:`E_s` (note the energy per bit
:math:`E_b` is just :math:`E_s/\log_2(M)`) to noise power spectral
density ratio, i.e., :math:`E_s/N_0` or :math:`E_b/N_0`. This allows
full BEP curves to be generated using just a single ensemble of ISI
patterns. The calculation of :math:`N_0`, taking into account the fact
that the total noise power is split between real/imagninary (or in 
digital communictions theory notation inphase/quadrature) parts is given by

.. math::
   :label: N0calc

   N_0 = \frac{E_s}{2\cdot 10^{(E_s/N_0)_\text{dB}/10}}

To be clear, :math:`(E_s/N_0)_\text{dB}` is the desired receiver
operating point. In the software simulation model we set
:math:`(E_b/N_0)_\text{dB}` convert to :math:`(E_s/N_0)_\text{dB}`,
arrive at :math:`N_0` for a fixed :math:`E_s`, then finally determine
:math:`\sigma_w`. Note the 2 in the denominator of
(:ref:`N0calc`) serves to split the total noise power between
the in-phase and quadrature components.

.. figure:: Block_SA.pdf
   :scale: 90%
   :align: center
   :figclass: htb

   Block diagram describing how for a linear channel from the WGN
   injection to the detector enable the use of semi-analytic BEP
   calculation. :label:`BlockSAsim`

The SA-BEP method first calculates the symbol error probability by
averaging over the ensemble of conditional Gaussian probabilities

.. math::
   :label: SABEP

   P_{E,\text{symb}} = \frac{1}{N} \sum_{k=1}^N \text{Pr}\{\text{Error}|z_k,
   \sigma_w,\text{other impairments}\}    

where :math:`N` is the number of symbols simulated to create the
ensemble. For the m-ary schemes MPSK and MQAM we assume is employed
[Ziemer]_, and the BEP values of interest are small
so we can write

.. math:: 
   :label: SEP2BEP

   \text{BEP} = \frac{P_{E,\text{symb}}}{\text{log}_2(M)}

The *other impairments* noted in (:ref:`SABEP`) refers to the
fact that SA-BEP can also be used to model carrier phase error or symbol
timing error.

For the SA-BEP analysis model what remains is to find expressions for
the conditional error probabilities in (4). A feature in the analysis of
both MPSK and MQAM, is that both schemes reside in a two dimensional
signal space and we can freely translate and scale signal points to a
*normalized location* to make the error probability equations easier to
work with.


M-ary PSK
---------

For MPSK with :math:`M > 2` the optimum decision region for symbol
detection is a wedge shaped region having interior angle :math:`\pi/M`,
as shown in the right side of Figure :ref:`saMPSK`. In [Ziemer]_ an 
exact SEP expression (and hence Gray coded BEP), attributed to [Craig]_, 
is given by:

.. math::
   :label: MPSKexact

   P_{E,\text{symb}} = \frac{1}{\pi}\int_0^{\pi-\pi/M} \exp\left(\frac{(E_s/N_0)
   \sin^2(\pi/M)}{\sin^2(\phi)}\right)\, d\phi

Avoiding numerical integration is desirable so that
(:ref:`SABEP`) can be computed quickly. A simple upper bound,
as described in [Ziemer]_ and
[Craig]_, considers the perpendicular distance
between the nominal signal space point following the matched filter and
the wedge shaped decision boundary as shown in
Figure :ref:`saMPSK`.

.. figure:: MPSK_SA_analysis.pdf
   :scale: 65%
   :align: center
   :figclass: htb

   Formulation of the conditional symbol error probability of MPSK
   (:math:`M=8` illustrated) given decision variable :math:`z_k`. :label:`saMPSK`


For unimpaired MPSK (no noise), we consider a normalized MPSK signal
point, :math:`z_k`, at angle zero to be :math:`(1,0)`. Since :math:`z_k`
is actually a complex baseband signal sample, it can be viewed as the
point :math:`z_k = 1 + j0` in the complex plane. The signal point length
being one corresponds to setting :math:`z_k = \sqrt{E_s} = 1`, where
:math:`E_s` is the symbol energy. The symbol error probability
:math:`P_{E,\text{symb}}` is over bounded by the probability of lying
above line :math:`L_a` or below line :math:`L_b`, when circularly
symmetric Gaussian noise is now added to :math:`z_k`. For the special
case of :math:`z_k = 1` the probabilities of being above and below the
lines are equal, hence this upper bound approximation results in

.. math::
   :label: MPSKbound

   P_{E,\text{symb}} \simeq 2Q\left(\frac{z_k\cdot\sin(\pi/M)}{\sigma_w}\right)=
   2Q\left(\frac{\sin(\pi/M)}{\sigma_w}\right),

where :math:`Q(x)` is the Gaussian :math:`Q` function given by

.. math::
   :label: Qfctn

   Q(x) = \frac{1}{\sqrt{2\pi}} \int_x^\infty e^{-t^2/2}\, dt

Since we have assumed that :math:`z_k = 1` we use :math:`\sigma_w` via
:math:`N_0` to control the operating point, :math:`E_s/N_0`, and hence
also :math:`E_b/N_0`. The over bound region, shown in light red in
Figure :ref:`saMPSK`, is due to double counting the error
probability in this region.

To demonstrate that this bound expression is adequate for the SA-BEP
modeling needs of this paper, we consider :math:`M=4` and 8 with
:math:`E_b/N_0` between 0 and 10 dB, focusing on BEP values above
:math:`10^{-3}`. Overlay plots of the exact BEP obtained from
(:ref:`MPSKexact`) and the bound of
(:ref:`MPSKbound`) are shown in
Figure :ref:`BEPMPSKcompare`.

.. figure:: 4PSK_8PSK_BEP_Exact_vs_Bound.pdf
   :scale: 65%
   :align: center
   :figclass: htb

   MPSK exact and bound BEP versus :math:`E_b/N_0` in dB for :math:`M=4`
   and 8. :label:`BEPMPSKcompare`

Only small differences are noted for the :math:`M=4` case, and then only
at very low :math:`E_b/N_0` values. The bound becomes tighter as
:math:`M` increases and as :math:`E_b/N_0` increases. We conclude that
the bounding expression for :math:`P_{E,\text{symb}}` is adequate for
use in semi-analytic BEP calculations at :math:`P_E` values below
:math:`10^{-3}`.

When matched filter mismatch is present the complex decision variable
:math:`z_k`, obtained by sampling the matched filter output, no longer
sits at a normalized value of :math:`(1,0) = 1\angle 0`. The scenario of
a perturbed :math:`z_k` is the real intent of
Figure :ref:`saMPSK`, where it shows two perpendicular
distances, :math:`d_a` and :math:`d_b`, for an arbitrary :math:`z_k`. We
now use these distances to form the conditional probability of symbol
error, and hence the gray coded BEP. Using simple geometry to write
:math:`d_a` and :math:`d_b` in terms of the angle :math:`\pi/M` and
:math:`z_k = |z_k|e^{j\theta_k}` we can finally write the conditional
symbol error probability as

.. math::
   :label: MPSKsepfnl
   :type: eqnarray

       P_{E,\text{symb}}(z_k,\sigma_w) &=& Q\left(\frac{|z_k|\sin(\pi/M - 
       |\theta_k|)}{\sigma_w}\right) + \nonumber \\
       && Q\left(\frac{|z_k|\sin(\pi/M + |\theta_k|)}{\sigma_w}\right).


M-ary Quadrature Amplitude Modulation
-------------------------------------

For MQAM the noise-free received symbols are scaled and translated to
lie nominally at :math:`(0,0)` in the complex plane. Here we pattern the
development of the SEP expression after Ziemer
[Ziemer]_. The decision region for correct symbol
detection detection is one of three types: (1) interior square, (2)
left/right or top/bottom channel to infinity, (3) corners upper
right/left and bottom right/left with two infinite sides, as depicted in
Figure :ref:`SAMQAM`.

.. figure:: MQAM_SA_analysis.pdf
   :scale: 65%
   :align: center
   :figclass: htb

   Formulation of the conditional symbol error probability of MQAM given
   decision variable :math:`z_k`. :label:`SAMQAM`


Using simplifications similar to the MPSK case, we have the following
equations for calculating the conditional SEP for symbol Types 1, 2, and
3. In the semi-analytic simulation software the symbol is known a
priori, so in forming the average of (:ref:`SABEP`) we choose
the appropriate expression. For type 1 we have:

.. math::
   :label: PEQAM1

   \begin{split}
       P_{E|\text{type 1}}(z_k,\sigma_w| \text{type 1}) \text{ = \hspace{1.45in}} \\
       Q\left(\frac{a - \text{Re}\{z_k\}}{\sigma_w}\right)
       + Q\left(\frac{a + \text{Re}\{z_k\}}{\sigma_w}\right) \\
       + Q\left(\frac{a - \text{Im}\{z_k\}}{\sigma_w}\right) 
       + Q\left(\frac{a + \text{Im}\{z_k\}}{\sigma_w}\right)
   \end{split}

For type 2 we have:

.. math::
   :label: PEQAM2

   \begin{split}
       P_{E|\text{type 2}}(z_k,\sigma_w| \text{type 2}) \text{ = \hspace{1.45in}} \\
       Q\left(\frac{a - \text{Re}\{z_k\}}{\sigma_w}\right) 
       + Q\left(\frac{a + \text{Re}\{z_k\}}{\sigma_w}\right) \\
       + Q\left(\frac{a \pm \text{Im}\{z_k\}}{\sigma_w}\right) 
   \end{split}

Finally for type 3 we have:

.. math::
   :label: PEQAM3

   \begin{split}
       P_{E|\text{type 3}}(z_k,\sigma_w| \text{type 3}) \text{ = \hspace{1.5in}} \\
       Q\left(\frac{a \pm \text{Re}\{z_k\}}{\sigma_w}\right)
       + Q\left(\frac{a \pm \text{Im}\{z_k\}}{\sigma_w}\right)
   \end{split}

In all three conditional probability of bit error expressions, (:ref:`PEQAM1`), 
(:ref:`PEQAM2`), and (:ref:`PEQAM3`), the variable :math:`a` is defined is defined in 
terms of the energy per symbol, :math:`E_s` and modulation order :math:`M` using

.. math:: 
   :label: QAMfinda

   a = \sqrt{\frac{3E_s}{2(M-1)}}.

Software Tools and Reproducible Science
---------------------------------------

All of the analysis and simulation software developed for this study is
written in Python. It makes use of the *scipy-stack* and the authors
GitHub project *scikit-dsp-comm* [Wickert1]_.
The code base specifics for this paper can be found on GitHub at
[Wickert2]_. The contents include Jupyter notebooks
and code modules. All of this is open-source and freely available.

Results
-------

Detailed performance scenarios are being compiled and will be presented
in both plots and tables. A sample of two result types: (A) without
equalization and (B) with equalization are given below.

Without Equalization
====================

Start with basic compare of 0.25/0.35 32MPSK. From a Jupyter notebook we
execute the following to produce the BEP performance plot of
Figure :ref:`32MPSK25to35` and obtain :math:`E_b/N_0` degradation in
dB at a BEP threshold of :math:`10^{-6}`:


.. figure:: BEP_32PSK_25to35_5d_plt.pdf
   :scale: 65%
   :align: center
   :figclass: htb
   

   32PSK BEP with 0.25/0.35 mismatch and no equalizer. :label:`32MPSK25t035`

The degradation is less than 0.5 dB for this rather small mismatch ratio
of 0.25/0.35. For a larger mismatch ratio the degradation will increase
rapidly, forcing the use of an equalizer should better knowledge of the
transmit excess bandwidth factor not be available at the receiver.

A tabular summary of mismatch losses for both MPSK and MQAM can be found respectively in Table :ref:`mismatchloss1` 
and Table :ref:`mismatchloss2`.
   
   
.. table:: MPSK degradation resulting from filter mismatch. :label:`mismatchloss1`
   :class: w
   :widths: auto

   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   |    | :math:`\mathbf{\alpha}_\text{tx}` | 0.25    | 0.25    | 0.25    | 0.25    | 0.25    | 0.3     | 0.35    | 0.4     | 0.45    | 0.5     |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   |    | :math:`\mathbf{\alpha}_\text{rx}` | 0.3     | 0.35    | 0.4     | 0.45    | 0.5     | 0.25    | 0.25    | 0.25    | 0.25    | 0.25    |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | M  | BEP                               | :math:`\hspace{1.9in} E_b/N_0` Degradation (dB)                                                   |
   +====+===================================+=========+=========+=========+=========+=========+=========+=========+=========+=========+=========+
   | 4  | :math:`10^{-5}`                   | 0\*     | 0\*     | 0\*     | 1.75E-2 | 3.82E-2 | 0\*     | 0\*     | 0\*     | 1.63E-2 | 3.78E-2 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 4  | :math:`10^{-6}`                   | 0\*     | 0\*     | 0\*     | 2.56E-2 | 4.70E-2 | 0\*     | 0\*     | 0\*     | 2.68E-2 | 4.88E-2 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 4  | :math:`10^{-7}`                   | 0\*     | 0\*     | 0\*     | 2.99E-2 | 5.64E-2 | 0\*     | 0\*     | 0\*     | 2.98E-2 | 6.36E-2 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 4  | :math:`10^{-8}`                   | 0\*     | 0\*     | 1.08E-2 | 3.54E-2 | 7.11E-2 | 0\*     | 0\*     | 1.28E-2 | 3.48E-2 | 7.28E-2 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 4  | :math:`10^{-9}`                   | 0\*     | 0\*     | 1.54E-2 | 4.56E-2 | 8.14E-2 | 0\*     | 0\*     | 1.60E-2 | 4.18E-2 | 8.10E-2 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 8  | :math:`10^{-5}`                   | 0\*     | 0\*     | 3.04E-2 | 7.82E-2 | 1.36E-1 | 0\*     | 0\*     | 3.37E-2 | 7.59E-2 | 1.42E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 8  | :math:`10^{-6}`                   | 0\*     | 0\*     | 4.01E-2 | 9.80E-2 | 1.88E-1 | 0\*     | 0\*     | 4.19E-2 | 1.04E-1 | 1.74E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 8  | :math:`10^{-7}`                   | 0\*     | 0\*     | 5.22E-2 | 1.17E-1 | 2.28E-1 | 0\*     | 0\*     | 4.37E-2 | 1.25E-1 | 2.12E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 8  | :math:`10^{-8}`                   | 0\*     | 1.04E-2 | 6.32E-2 | 1.35E-1 | 2.55E-1 | 0\*     | 0\*     | 6.11E-2 | 1.48E-1 | 2.59E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 8  | :math:`10^{-9}`                   | 0\*     | 1.52E-2 | 7.11E-2 | 1.62E-1 | 3.08E-1 | 0\*     | 1.26E-2 | 6.77E-2 | 1.66E-1 | 3.01E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 16 | :math:`10^{-5}`                   | 0\*     | 3.27E-2 | 1.35E-1 | 3.20E-1 | 6.54E-1 | 0\*     | 3.73E-2 | 1.38E-1 | 3.10E-1 | 5.49E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 16 | :math:`10^{-6}`                   | 0\*     | 4.71E-2 | 1.74E-1 | 4.04E-1 | 7.12E-1 | 0\*     | 4.27E-2 | 1.70E-1 | 4.14E-1 | 6.76E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 16 | :math:`10^{-7}`                   | 0\*     | 5.85E-2 | 2.40E-1 | 4.78E-1 | 8.79E-1 | 0\*     | 5.76E-2 | 2.20E-1 | 4.68E-1 | 8.35E-1 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 16 | :math:`10^{-8}`                   | 1.27E-2 | 6.48E-2 | 2.33E-1 | 5.97E-1 | 1.07E+0 | 1.02E-2 | 6.27E-2 | 2.49E-1 | 5.92E-1 | 1.04E+0 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 16 | :math:`10^{-9}`                   | 1.19E-2 | 7.31E-2 | 2.90E-1 | 6.28E-1 | 1.15E+0 | 1.44E-2 | 8.18E-2 | 2.88E-1 | 6.68E-1 | 1.25E+0 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 32 | :math:`10^{-5}`                   | 3.63E-2 | 1.69E-1 | 5.82E-1 | 1.49E+0 | 2.60E+0 | 3.62E-2 | 1.69E-1 | 5.81E-1 | 1.36E+0 | 2.54E+0 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 32 | :math:`10^{-6}`                   | 4.95E-2 | 2.01E-1 | 7.07E-1 | 1.91E+0 | 3.22E+0 | 4.80E-2 | 1.94E-1 | 7.02E-1 | 1.55E+0 | 2.96E+0 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 32 | :math:`10^{-7}`                   | 5.67E-2 | 2.48E-1 | 9.17E-1 | 1.95E+0 | 4.40E+0 | 6.51E-2 | 2.50E-1 | 9.25E-1 | 2.09E+0 | 3.83E+0 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 32 | :math:`10^{-8}`                   | 7.35E-2 | 2.90E-1 | 9.56E-1 | 2.38E+0 | 4.82E+0 | 8.07E-2 | 3.40E-1 | 1.11E+0 | 2.59E+0 | 4.07E+0 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | 32 | :math:`10^{-9}`                   | 8.62E-2 | 3.18E-1 | 1.15E+0 | 2.79E+0 | 5.27E+0 | 8.26E-2 | 3.29E-1 | 1.19E+0 | 3.25E+0 | 4.37E+0 |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
   | \* degradation less than 0.01 dB                                                                                                           |
   +----+-----------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+


.. table:: MQAM degradation resulting from filter mismatch. :label:`mismatchloss2`
   :class: w
   :widths: auto

   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   |     | :math:`\mathbf{\alpha}_\text{tx}` | 0.25     | 0.25     | 0.25    | 0.25    | 0.25    | 0.3      | 0.35     | 0.4     | 0.45    | 0.5     |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   |     | :math:`\mathbf{\alpha}_\text{rx}` | 0.3      | 0.35     | 0.4     | 0.45    | 0.5     | 0.25     | 0.25     | 0.25    | 0.25    | 0.25    |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | M   | BEP                               | :math:`\hspace{2.0in} E_b/N_0` Degradation (dB)                                                       |
   +=====+===================================+==========+==========+=========+=========+=========+==========+==========+=========+=========+=========+
   | 4   | :math:`10^{-5}`                   | 0\*      | 0\*      | 0\*     | 2.36E-2 | 4.85E-2 | 0\*      | 0\*      | 0\*     | 2.31E-2 | 4.58E-2 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 4   | :math:`10^{-6}`                   | 0\*      | 0\*      | 0\*     | 2.45E-2 | 4.64E-2 | 0\*      | 0\*      | 0\*     | 2.61E-2 | 4.90E-2 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 4   | :math:`10^{-7}`                   | 0\*      | 0\*      | 0\*     | 2.46E-2 | 4.91E-2 | 0\*      | 0\*      | 0\*     | 2.20E-2 | 4.83E-2 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 4   | :math:`10^{-8}`                   | 0\*      | 0\*      | 0\*     | 2.32E-2 | 4.63E-2 | 0\*      | 0\*      | 0\*     | 2.52E-2 | 4.53E-2 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 4   | :math:`10^{-9}`                   | 0\*      | 0\*      | 0\*     | 2.56E-2 | 4.85E-2 | 0\*      | 0\*      | 0\*     | 2.28E-2 | 4.71E-2 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 16  | :math:`10^{-5}`                   | -1.10E-2 | 5.54E-2  | 1.54E-2 | 1.75E-1 | 2.83E-1 | -4.31E-2 | 4.07E-2  | 5.75E-3 | 1.16E-1 | 2.80E-1 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 16  | :math:`10^{-6}`                   | 3.99E-2  | -1.40E-2 | 9.43E-2 | 7.97E-2 | 2.38E-1 | 0\*      | 2.56E-2  | 3.66E-2 | 1.91E-1 | 2.26E-1 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 16  | :math:`10^{-7}`                   | -2.76E-2 | 1.75E-2  | 7.63E-2 | 2.05E-1 | 2.91E-1 | 5.10E-2  | -2.44E-2 | 5.85E-2 | 2.18E-1 | 2.53E-1 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 16  | :math:`10^{-8}`                   | -2.63E-2 | 3.09E-2  | 3.55E-2 | 1.36E-1 | 2.73E-1 | 3.11E-2  | 2.15E-2  | 5.17E-3 | 1.31E-1 | 2.54E-1 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 16  | :math:`10^{-9}`                   | 3.39E-2  | -1.29E-2 | 4.19E-3 | 8.62E-2 | 2.22E-1 | 1.91E-2  | 7.10E-3  | 8.02E-2 | 1.27E-1 | 2.22E-1 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 64  | :math:`10^{-5}`                   | -6.12E-2 | 3.31E-2  | 2.45E-1 | 6.80E-1 | 1.19E+0 | -2.93E-2 | 1.43E-1  | 2.81E-1 | 5.53E-1 | 1.17E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 64  | :math:`10^{-6}`                   | 4.07E-2  | 9.64E-2  | 2.63E-1 | 6.48E-1 | 1.33E+0 | 4.75E-2  | 4.16E-2  | 2.63E-1 | 8.29E-1 | 1.14E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 64  | :math:`10^{-7}`                   | 3.47E-4  | 1.13E-1  | 3.13E-1 | 5.22E-1 | 1.14E+0 | -2.16E-2 | 5.49E-2  | 3.34E-1 | 7.15E-1 | 1.18E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 64  | :math:`10^{-8}`                   | 1.91E-2  | 1.43E-1  | 3.07E-1 | 7.02E-1 | 1.20E+0 | 1.09E-2  | 5.43E-2  | 3.06E-1 | 7.47E-1 | 1.18E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 64  | :math:`10^{-9}`                   | 2.23E-2  | -4.09E-2 | 3.23E-1 | 6.46E-1 | 1.16E+0 | 8.26E-3  | 9.10E-2  | 3.21E-1 | 6.80E-1 | 1.15E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 256 | :math:`10^{-5}`                   | 9.59E-2  | 2.72E-1  | 1.36E+0 | 3.12E+0 | 6.90E+0 | 1.24E-1  | 3.25E-1  | 1.29E+0 | 3.20E+0 | 5.47E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 256 | :math:`10^{-6}`                   | 6.83E-2  | 3.18E-1  | 1.26E+0 | 3.33E+0 | 7.01E+0 | 1.68E-1  | 3.27E-1  | 1.34E+0 | 3.53E+0 | 7.32E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 256 | :math:`10^{-7}`                   | 5.71E-2  | 3.36E-1  | 1.46E+0 | 3.25E+0 | 6.59E+0 | 3.21E-2  | 3.94E-1  | 1.29E+0 | 3.37E+0 | 6.90E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 256 | :math:`10^{-8}`                   | 3.89E-2  | 3.39E-1  | 1.17E+0 | 3.46E+0 | 8.66E+0 | -3.30E-2 | 3.31E-1  | 1.35E+0 | 3.18E+0 | 7.37E+0 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | 256 | :math:`10^{-9}`                   | 9.90E-2  | 4.04E-1  | 1.34E+0 | 3.68E+0 | 6.79E+0 | 5.65E-2  | 3.67E-1  | 1.36E+0 | 3.36E+0 | 1.10E+1 |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+
   | \* degradation less than 0.01 dB                                                                                                                |
   +-----+-----------------------------------+----------+----------+---------+---------+---------+----------+----------+---------+---------+---------+

With Constrained Use of Equalization
====================================

Continue from previous subsection but bring in 11-tap and 21 tap
equalizers. Consider also the setting of the LMS convergence parameter
:math:`\mu` and the training duration. Again we show the Python code
that runs in Jupyter notebook to model pulse shape mismatch and this
time implements a short tap length adaptive equalizer to mitigate the
BEP degradation. The gradation with and without equalization is shown in
the listing below:
|

The BEP plot shown in Figure :ref:`16QAM25to50` demonstrates how a
21-tap equalizer can mitigate the mismatch losses when the excess
bandwidth factor ratio is rather large, i.e., 0.25/0.5.

.. figure:: BEP_16QAM_25to5_eq21.pdf
   :scale: 65%
   :align: center
   :figclass: htb
   

   16QAM BEP with large mismatch and then the inclusion of a 21-tap
   equalizer. :label:`16QAM25to50`


Conclusion and Planned Extensions
---------------------------------

A few results are included in this draft, but more will be provided in
the final version.

Planned extensions include degradations due to phase jitter, static
phase error, and timing errors.


Acknowledgment
--------------

The author wishes to thank Jim Rasmussen for generating interest in this
topic and related discussions that have taken place over the last few
years working at Cosmic AES.

References
----------

.. [Lindsey] W. Lindsey and M. Simon, Telecommunications Systems Engineering, original edition Prentice Hall, 1973. Reprint Dover Publications, 2011.

.. [Ziemer] R. Ziemer and W. Tranter, Principles of Communications, seventh edition, John Wiley, 2015. 

.. [Proakis] G.J. Proakis, Digital Communications, 4th ed., McGraw Hill, 2001.

.. [Goldsmith] A. Goldsmith, Wireless Communications, Cambridge University Press, 2005.

.. [Tranter] W. Tranter, K. Shanmugan, T. Rappaport, and K. Kosbar, Principles of Communication Systems Simulation with Wireless Applications, Prentice Hall, 2004.

.. [Harris] F. Harris, C. Dick, S. Seshagiri, and K. Moerder, “An improved square-root nyquist shaping filter,” Proceeding of the SDR 05 Technical Conference and Product Exposition, 2005.

.. [Xing] T. Xing, Y. Zhan, and J. Lu, “A Performance Optimized Design of Receiving Filter for Non-Ideally Shaped Modulated Signals,” in *IEEE International Conference on Communications*, p. 914-919, 2008.

.. [Rappaport] T. Rappaport, Wireless Communications: Principles and Practice, Prentice Hall, 1999.

.. [Craig] J. Craig, “A New, Simple and Exact Result for Calculating the Probability of Error for Two-Dimensional Signal Constellations,” in *IEEE Milcom ’91*, p. 571-575, 1991.

.. [Wickert1] M. Wickert, “Scikit-dsp-comm: a collection of functions and classes to support signal processing and communications theory teaching and research,” https://github.com/mwickert/scikit-dsp-comm. 

.. [Wickert2] M. Wickert, “Matched filter mismatch losses: a Python sofware repository”, https://github.com/mwickert/Matched_Filter_Mismatch_Losses.

.. [Fitzpatrick] Fitzpatrick, W., Wickert, M., and Semwal, S. (2013) 3D Sound Imaging with Head Tracking, *Proceedings IEEE 15th Digital Signal Processing Workshop/7th Signal Processing Education Workshop*.

.. [Beranek] Beranek, L. and Mellow, T (2012). *Acoustics: Sound Fields and Transducers*. London: Elsevier.

.. _`https://github.com/mwickert/scikit-dsp-comm`: https://github.com/mwickert/scikit-dsp-comm

.. _`https://github.com/mwickert/Matched_Filter_Mismatch_Losses`: https://github.com/mwickert/Matched_Filter_Mismatch_Losses

.. _`10.25080/Majora-4af1f417-00e`: http://conference.scipy.org/proceedings/scipy2018/mark_wickert_250.html