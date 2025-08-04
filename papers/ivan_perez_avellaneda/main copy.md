---
# Ensure that this title is the same as the one in `myst.yml`
#title: 'Paper: SciPy Optimize and Reachability of Nonlinear Systems'
title: 'CFSpy: A Python Library for the Computation of Chen-Fliess Series'
abstract: |
  In this paper, CFSpy a package in Python that numerically computes Chen-Fliess series
  is presented. The reachable sets of non-linear systems are
  calculated with this package. The method used obtains batches of iterative integrals
  of the same length instead one at a time. For this, we consider the alphabetical
  order of the words that index the series. By redefining the iterative
  integral reading the index word in the opposite direction from right
  to left, we allow the broadcasting of the computation of the iterative
  integral to all permutations of a length. Assuming the input
  is sufficiently well approximated by piece-wise step functions in 
  a fine partition of the time interval, the minimum bounding box of a 
  rechable set is computed by means of polynomials in terms of the inputs. To solve the optimization problem, the SciPy library is used.

---

## Introduction


Control systems describe the dynamics of mechanisms driven by an input vector. 
In engineering, many of these systems are affected by disturbances or are complex
enough that a simplified version of the model is used instead. This endows the
systems with uncertainty and impairs their safety operations.
The reachable set is a tool that helps analyse safety-related properties.
For a given final time, it is defined as the set of all outputs 
as a response of the system to a given set of initial inputs and states.
The overestimation of the reachable set is associated to safety and obstable avoidance
and the underestimation to the *liveness* of the system [@Chen2015;@Seo2019]. 

Different methodologies are used to compute the reachable set or an approximation of it.
Among the most popular techniques are the Hamilton-Jacobi framework (HJ) [@Bansal2017;@Mitchell2007;@Mitchell2005] 
which uses game theory between
two players where one player drives the system away from
the goal while the other moves it towards the goal, contraction-based which uses contraction theory
[@Maidens2015;@Bullo2022] on the Jacobian of the vector field 
of the system, set-based that uses different set representations,
monotone systems [@Scott2013;@Meyer2019;@Jafarpour2024], and mixed-monotonicity [@Coogan2020]. 
There is also the simulation-based rechability [@Fan2017;@Huang2012]
and recently Chen-Fliess series have been used for this purpose. 

An important problem in the computation of the 
reachable set is the *curse of dimensionality*. 
This consists in the increasing of the complexity of the algorithm
as the dimension increases. 
To tackle this in HJ-based methods, in @He2024,
the authors use a new approach that requires
the definition of an admissible control set 
to provide control policies that are 
consistent on the coupled subsystems. This fixes an incosistency
issue with the controls called the *leaking corner*. 
For set-based methods, the use of zonotopes 
is known to have low complexity [@Althoff2016].

%In [@Qasem2024] the inverted pendulum is described as control system taking ... as the inputs
%and in [@He2024], the planar vertical take-off and landing (PVTOL) which describes
%a simplified unmanned aircraft vehicle (UAV) model is used to analyze the reachability of
%the system.

A Chen-Fliess (CFS) series provides a local representation of the output of 
a non-linear control-affine system in terms of its input [@Fliess81]. 
Given its coefficients, the series overlooks the dynamics when computing the output. 
This is important in cases where the system is unknown or 
affected by uncertainty. Then the coefficients are learned by
using online learning methods [@Venkatesh2019]. In @Perez2022, a version of noncommutative differential calculus
was developed to represent the derivative of a CFS. 
This was used to perform reachability analysis by applying
second degree optimization of CFS in @Perez2023.

In the present work, CFSpy [@Perez2024], a Python library for the computation of Chen-Fliess series is introduced
and it is used to perform reachability analysis of non-linear system by
assuming the input functions can be piece-wise approximated. A polynomial
of the Chen-Fliess series is obtained and then optimized to get the
points of an overstimation of the reachable set. 
The SciPy package [@scipy] is used to performed the optimization.

The outline of the paper is the following: in [](#sec_preliminaries),
the preliminary concepts and results in reachability, formal languages and CFS
are presented. In [](#sec_results), the algorithms for the computation of
the iterative integrals and the lie derivatives are provided which are 
the components of CFS. Then, the numerical computation of CFS is shown
and examples are given. Finally, in [](#sec_conclusions), the conclusions and
future work are described.


(sec_preliminaries)=
## Preliminaries 

In the current section,  we give the definitions and results needed to explain
our main contribution: the CFSpy Python package. For this, in [](#sec_matrixop), 
tools from linear algebra and matrix operations are presented to explain the algorithms. 
In [](#sec_language), concepts from formal language
theory are presented to characterize CFS which are defined in [](#sec_CFS). These provide a
representation of the output of a nonlinear-affine systems in terms of iterated integrals
of the input.

(sec_matrixop)=
### Matrix Operations

As we will see in [](#sec_results), to avoid computing the components of the Chen-Fliess series
one by one, this is, permutation by permutation, matrix operations such as the Kronecker product
to represent the stacking of two matrices and the elementwise multiplication or Hadamard product
 are defined. 

:::{prf:definition} Kronecker Product
:label: def_kronecker

Given the matrix $A_{m,n}$ and $B_{p,q}$, the Kronecker product $A\otimes B$ is defined by

$$
A\otimes B = 
\begin{bmatrix}
  a_{11}B & \cdots & a_{1n}B\\
  \vdots & \ddots & \vdots\\
  a_{m1}B & \cdots & a_{mn} B
\end{bmatrix}
$$
:::

In particular, stacking a matrix $B$ vertically $k$ times is written in terms of this product and the unitary 
vector as follows

:::{prf:example}
:label: ex_kronecker
Denote $\mathbb{1}_k$ the vector of $k$ ones and take the arbitrary matrix $B$, then


```{math}
:label: 
\mathbb{1}_{k}\otimes B = 
\begin{bmatrix}
B \\
\vdots \\
B
\end{bmatrix}
```
:::



The next operation is useful to perfom the iterative integration
of the inputs represented as matrices.

:::{prf:definition} Hadamard Product
:label: def_hadamard

Given the matrix $A_{m,n}$ and $B_{m,n}$, the Hadamard product is defined by

$$
(A\odot B)_{ij} = (A)_{ij}(B)_{ij}
$$
:::


:::{prf:example}
:label: ex_hadamard
Consider the matrices $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$ 
and $B = \begin{bmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix}$. 
The Hadamard product $A \odot B$ is the following

```{math}
:label:
A\odot B = 
\begin{bmatrix}
7 & 16 & 27 \\
40  & 55 & 72
\end{bmatrix}
```
:::

The next operation characterizes the vertical stacking of two 
different matrices.

:::{prf:definition} Vertical Direct Sum
:label: def_directsum

Given the matrix $A_{m,n}$ and $B_{r,n}$, the vertical direct sum is defined by

$$
A\oplus_v B = 
\begin{bmatrix}
A\\
B
\end{bmatrix}
$$
:::


:::{prf:example}
:label: ex_vertical_stack
Consider the matrices $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$ 
and $B = \begin{bmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \\ 13 & 14 & 15 \end{bmatrix}$. 
The vertical direct sum $A \oplus_{v} B$ is the following

```{math}
:label:
A\oplus_{v} B = 
\begin{bmatrix}
1 & 2 & 3 \\
4  & 5 & 6 \\
7 & 8 & 9 \\
10 & 11 & 12 \\
13 & 14 & 15 \\
\end{bmatrix}
```
:::



(sec_language)= 
### Formal Language Theory

The CFS is indexed by words of any length.
Formally, these words are elements
of an algebraic structure called *free monoid*
where the *alphabet* is a subset that along with
the *concatenation* operation act as the generator.
Words are the noncommutative counterpart of monomials 
and they extend as the basis of polynomials and then to power series.
These are important in the theory of CFS since 
power series are isomorphic to CFS.
In the present section, concepts of
formal language theory [@Salomaa1973] are presented. 


:::{prf:definition} Monoid
:label: def_monoid

The tuple $(S, \cdot, e)$ is a *monoid* if
the operation $\cdot: S\times S \rightarrow S$ satisfies the associativity property
and $e$ is the identity element under $S$.
:::

To make notation concise, given the monoid $(S,\cdot)$, we write $\cdot(s_1, s_2)$ as
the concatenation $s_1 s_2$
then by the associativity $(s_1 s_2) s_3 = s_1 (s_2 s_3) = s_1 s_2 s_3$. Thus,
the operation $\cdot: S\times S \rightarrow S$ is referred to as *concatenation*.
The concatenation of finite number of elements of $S$ is called a *word*. 


A *free monoid* $(X^*, \cdot, \empty)$ generated 
by the set $X$ is the monoid of all finite concatenations of the elements of $X.$
 The generating set $X$ is called *alphabet* and its elements
 *letters*. Next, we define a function that helps us classify words and
define the algorithms in [](#sec_results).

:::{prf:definition} Length of a Word
:label: def_length

Given the free monoid $(X^*, \cdot, \empty)$ with alphabet $X = \{x_0, \cdots, x_m\}$,
the *length* of a word is the function $|\cdot |: X^* \rightarrow \mathbb{N}$ such that 
for an arbitrary word $x_{i_1}\cdots x_{i_n} \in X^*$, it assigns
$$
|x_{i_1}\cdots x_{i_n}| = n
$$
:::

The set of all words of length $k$ is written as $X^k$. Then, we can express $X^* = \cup_{k=0}^{\infty}X^k$.
Next, we define the set of formal power series over $\mathbb{R}$ with indeterminates in $X^*$.

:::{prf:definition} Formal Power Series
:label: def_freealg

A *formal power series* $c$ with indeterminates in $X^*$ and coefficients $(c,\eta) \in \mathbb{R}^\ell$
has the form:

```{math}
c := \sum_{\eta \in X^*} (c,\eta) \eta
```

:::

The set of all power series $c$ is denoted $\mathbb{R}^\ell\langle \langle X\rangle \rangle$ and is extended to
an *algebra* structure by adding the the *shuffle* operator.


(sec_CFS)=
### Chen-Fliess Series

CFS has its roots in the works of [@Chen1957] and [@Fliess81]
and provide an input-output representation of nonlinear control-affine
system. They are defined in terms of iterated integrals.
To identify them with a system, the coefficients are written
in terms of Lie derivatives of the vector field. In the current section,
The CFS are presented.

By a nonlinear control-affine system, we refer to the following
set of equations:

$$
\label{nonlinsys}
\dot{z} &= g_0(z) + \sum_{i = 1}^{m} g_i(z) u_i\\
y &= h(z)
$$
where $z = (z_1, \cdots, z_n)$ is the vector state of the system, $u= (u_1, \cdots, u_m)$ is the control input vector of the system and
$y$ is the output.

The iterated integral associated with the word index $\xi = x_{i_1}, \cdots, x_{i_r}$
maps each letter of an alphabet with a coordinate of the vector input function and integrates
recursively each input coordinate in the order of the letters in the given word. Specifically, we have
the following:


:::{prf:definition}
:label: def_iterint

Given the free monoid $(X^*, \cdot,\empty)$, and the word $\xi = x_{i}\eta \in X^*$, the *iterative integral*
of $u\in L^m[0, T]$ associated to $\xi$ is the operator $E_{\xi}:L^m[0, T]\rightarrow C[0,T]$, described
recursively by

$$
\begin{align*}
E_{\empty}[u](t) = 1
\end{align*}
$$

$$
\begin{align*}
E_{x_{i}\eta}[u](t) = \int_{0}^{t}u_{x_{i}}(\tau)E_{\eta}[u](\tau)d\tau
\end{align*}
$$
:::

The definition of the iterated integral associated to a word
is naturally extended to power series in the following manner:

:::{prf:definition}
:label: def_cfs

Given the free monoid $(X^*, \cdot,\empty)$, and the formal power series $c\in \mathbb{R}^\ell\langle\langle X\rangle\rangle$, the *Chen-Fliess series* associated to $c$ is the functional, $F_{c}[\cdot](t):L^p[0, T]\rightarrow \mathbb{R}^\ell$, described by

$$
\begin{align*}
F_c[u](t) = \sum_{\eta \in X^*} (c, \eta) E_{\eta}[u](t)
\end{align*}
$$
:::

The support $\text{sup}(c)$ of a formal power series $c= \sum_{\eta\in X^\ast}(c,\eta)\eta$ is the set of words $\eta\in X^\ast$
that have non-null associated coefficients. This is, $(c,\eta)\neq 0$. Computationally, we work with 
power series of a finite support. Specifically,
CFS truncated to words of length $N$ denoted as $F^N_c[u](t)$. This is, the formal power series $c$ of the CFS is truncated to words
of length $N$ and we have

$$
\begin{align*}
F^N_c[u](t) = \sum_{k \leq N}\sum_{\eta \in X^k} (c, \eta) E_{\eta}[u](t).
\end{align*}
$$

Nest, to provide the association of the CFS with a nonlinear control-affine system,
we give the defintion of a Lie derivative.


:::{prf:definition}
:label: def_liederiv

Given the system in [](#nonlinsys), the Lie derivative associated to the word $\eta \in X^*$ of the function $h:\mathbb{R}^n\rightarrow \mathbb{R}$  is the following:

$$
\begin{align*}
L_{\eta}h = L_{x_{i_1}}\cdots L_{x_{i_k}}h
\end{align*}
$$

where we have that $L_{x_j}h = \left(\frac{\partial}{\partial x}h\right) \cdot g_j$
:::

The following result gives the conditions under which a nonlinear system
has an input-output representation by CFS.

:::{prf:theorem} Fliess, 1983
:label: th_nonlsys_rep

Consider the system in [](#nonlinsys). 
The Chen-Fliess series $F_c$ represents the system
if and only if

* The coefficient of the Fliess operator satisfies: $(c,\eta) = L_{\eta}h(z)|_{z_0}$ for $\eta\in X^*$.
* The power series $c$ has finite Lie rank.
* There exist $K, M\geq 0$ such that $|(c,\eta)|\leq KM^{|\eta|}|\eta|!$

:::


(sec_results)=
## Main Results

In the current section, the numerical computation of the Chen-Fliess series is addressed.
For this, the iterative integral and the Lie derivative are redefined to allow for
easier algorithm implementations. In [](#sec_CFS), they are written recursively.  In this section, we changed
the direction of the expressions to compute them iteratively. 
Instead of looking at the definition of the components forwardly, 
they are rewritten backwardly. 

Another aspect of the numerical computation that we describe is how to avoid calculating 
each component of the Chen-Fliess series word by word. For this, the new definitions are
broadcasted to generate batches of a fixed length that will be stacked and multiplied
componentwise to obtain the final output. This increases the speed of the algorithms.



(sec_broadcasting)=
### Broadcasting


Consider the partition $\mathcal{P} = \lbrace t_0, \cdots, t_R\rbrace$ of the time interval $[0, T]$ 
and the input vector function $u = (u_0, \cdots, u_m) \in L^{m+1}_p[0,T]$. To avoid computing each iterative integral
one word at the time and, instead, obtain the batch of all words of a fixed length, we need to consider
the alphabetical order of the words. First, broadcast the integration of the input function 

```{math}
\int u = \left(\int u_0, \cdots, \int u_m\right),
```
then broadcast the multiplication by the first component $u_0$ and 
integrate, this is, 
```{math}
\int u_0\int u = \left(\int u_0\int u_0, \cdots, \int u_0\int u_m\right).
```
Repeat the process 
for each of the components and stack the result, we obtain the array 
```{math}
:label: eq_block_ethree
\int u\int u = \left[\int u_i\int u_0 \cdots \int u_i\int u_m\right]_{i\in \lbrace 1, \cdots, m\rbrace}.
```
Notice that $\int u \int u$ is the array
of iterated integral $E_{\eta}[u]$ of words $\eta$ of length two. Inductively, given the array of iterated integrals 
of all words in $X^k$ and denote it 

```{math}
\left(\int u\right)^k
```
we construct the array of iterated integral of words in $X^{k+1}$ by broadcasting the multiplication of
each component $u_i$ and the integration. This is,

```{math}
\left(\int u\right)^{k+1} = \left[ \int u_i\odot\left(\int u\right)^k\right]_{i\in \lbrace 1, \cdots, m\rbrace}
```

Numerically, the following steps are followed. Take the matrix of the partitioned input functions

```{math}
\begin{bmatrix}
u_0(t_0) & \cdots & u_0(t_{R})\\
\vdots & \ddots & \vdots\\
u_m(t_0) & \cdots & u_m(t_{R})
\end{bmatrix}

```
integrate by taking the cummulative each component of the matrix and multiplying
by a given rectangle size $\Delta$


```{math}
\begin{bmatrix}
\sum_{i = -1}^{0} u_0(t_i) & \cdots & \sum_{i = -1}^{R} u_0(t_i)\\
\vdots & \ddots & \vdots\\
\sum_{i = -1}^{0} u_m(t_i) & \cdots & \sum_{i = -1}^{R} u_m(t_i)
\end{bmatrix}
\Delta
```
broadcast the multiplication of the first component $u_0$
```{math}
\begin{bmatrix}
u_0(t_0) & \cdots & u_0(t_R)\\
\vdots & \ddots & \vdots\\
u_0(t_0) & \cdots & u_0(t_R)
\end{bmatrix}_{(m,\ T+1)}
\odot
\begin{bmatrix}
\sum_{i = -1}^{0} u_0(t_i) & \cdots & \sum_{i = -1}^{R} u_0(t_i)\\
\vdots & \ddots & \vdots\\
\sum_{i = -1}^{0} u_m(t_i) & \cdots & \sum_{i = -1}^{R} u_m(t_i)
\end{bmatrix}
\Delta
```
repeat for each component of the input and stack them as in equation [](#eq_block_ethree)
to obtain numerically the list of $E_{\eta}[u]$ for $\eta\in X^3$

```{math}
\begin{bmatrix}
\sum_{k=-1}^0 u_0(t_k)\left(\sum_{i = -1}^{k} u_0(t_i)\right)\Delta & \cdots & \sum_{k = -1}^R u_0(t_k)\left(\sum_{i = -1}^{k} u_0(t_i)\right)\Delta\\
\vdots & \ddots & \vdots\\
\sum_{k=-1}^0 u_0(t_k)\left(\sum_{i = -1}^{k} u_m(t_i)\right)\Delta & \cdots & \sum_{k = -1}^R u_0(t_k)\left(\sum_{i = -1}^{k} u_m(t_i)\right)\Delta\\
\\
\hdashline\\
& \vdots\\
\\
\hdashline
\\
\sum_{k=-1}^0 u_m(t_k)\left(\sum_{i = -1}^{k} u_0(t_i)\right)\Delta & \cdots & \sum_{k = -1}^R u_m(t_k)\left(\sum_{i = -1}^{k} u_0(t_i)\right)\Delta\\
\vdots & \ddots & \vdots\\
\sum_{k=-1}^0 u_m(t_k)\left(\sum_{i = -1}^{k} u_m(t_i)\right)\Delta & \cdots & \sum_{k = -1}^R u_m(t_k)\left(\sum_{i = -1}^{k} u_m(t_i)\right)\Delta\\
\end{bmatrix}
\Delta
```



(sec_compiterint)=
### Computation of Iterated Integrals


The following provides a different view of [](#def_iterint) that we use later to
compute the iterated integrals.

:::{prf:definition}
:label: def_newiterint

Consider the word $\eta = x_{i_1}\cdots x_{i_{r}} \in X^*$, the *backward* iterative integral
of $u\in L^p[0, T]$ associated to $\eta$ is the operator $H_{\eta}(\cdot)$ described
recursively by

```{math}
H_{x_{i_1}}(H_{x_{i_2}}(\cdots H_{x_{i_r}}))
```
where 

```{math}
H_{x_{i_j}}(\cdot) =  
  \begin{cases}
   \quad \displaystyle\int u_{i_j} &\text{if } j = r, \\
   \quad \displaystyle\int (\cdot) &\text{if } j = 0, \\
   u_{i_j}\displaystyle\int(\cdot ) &\text{otherwise. } 
  \end{cases}

```
:::


The following algorithm provides the matrix of the stacked iterated integrals.
This is the list of iterated integrals associated with words of length less than 
a certain number $N$.

:::{prf:algorithm} 
:label: matrix_iter_int

**Inputs** Given the truncation length $N$, the inputs $u$ of the system in [](#nonlinsys)   

**Output** The matrix $\mathcal{U}$ of the stacking of iterative integrals $E_\eta[u](t)$ for $|\eta|\leq N$

$U_0 \leftarrow 1 \oplus_v u$

$U_1 \leftarrow [\textbf{0}\ |\ S(U_0)_{:,\hat{T}}\Delta]$

$\mathcal{U} \leftarrow U_1$

For $k$ in $\{1,\cdots, N-1\}$ do:
>    1. $V \leftarrow \textbf{1}_{m} \otimes U_k$ 
>    2. $v \leftarrow [I_{m}\otimes \textbf{1}_{N_{U_{k}}}] u$ 
>    3. $M \leftarrow v\odot V$
>    4. $U_{k+1} \leftarrow [\textbf{0}\ |\ S(M)_{:,-\hat{T}}\Delta]$
>    5. $\mathcal{U} \leftarrow U_k \oplus_{v} \mathcal{U}$
:::


The Python code of the above algorithm is the following:

```{code} python
:label: my-program
:caption: Iterated Integrals
def iter_int(u,t0, tf, dt, Ntrunc):
    import numpy as np

    # The length of the partition of time is computed
    length_t = int((tf-t0)//dt+1)
    
    if u.shape[1] != length_t:
        raise ValueError("The length of the input, %s, must be int((tf-t0)//dt+1) = %s." %(u.shape[1], length_t))

    t = np.linspace(t0, tf, length_t)
    # The input u_0 associated with the letter x_0 is generated.
    u0 = np.ones(length_t)

    u = np.vstack([u0, u])
    
    # The number of rows which are equal to the number of total input functions is obtained.
    num_input = int(np.size(u,0))

    total_iterint = num_input*(1-pow(num_input,Ntrunc))/(1-num_input)
    # This is transformed into an integer.
    total_iterint = int(total_iterint)
    
    Etemp = np.zeros((total_iterint,length_t))

    ctrEtemp = np.zeros(Ntrunc+1)

    for i in range(Ntrunc):
        ctrEtemp[i+1] = ctrEtemp[i]+pow(num_input,i+1)

    sum_acc = np.cumsum(u, axis = 1)*dt

    Etemp[:num_input,:] = np.hstack((np.zeros((num_input,1)), sum_acc[:,:-1]))

    for i in range(1,Ntrunc):
        # start_prev_block = num_input + num_input**2 + ... + num_input**(i-1)
        start_prev_block = int(ctrEtemp[i-1])
        # end_prev_block = num_input + num_input**2 + ... + num_input**i
        end_prev_block = int(ctrEtemp[i])
        # end_current_block = num_input + num_input**2 + ... + num_input**(i+1)
        end_current_block = int(ctrEtemp[i+1])
        # num_prev_block = num_input**i
        num_prev_block = end_prev_block - start_prev_block
        # num_current_block = num_input**(i+1)
        num_current_block = end_current_block - end_prev_block
        
        U_block = u[np.repeat(range(num_input), num_prev_block), :]
        
        prev_int_block = np.tile(Etemp[start_prev_block:end_prev_block,:],(num_input,1))
  
        current_int_block = np.cumsum(U_block*prev_int_block, axis = 1)*dt
        # Stacks the block of iterated integrals of word length i+1 into Etemp
        Etemp[end_prev_block:end_current_block,:] = np.hstack((np.zeros((num_current_block,1)), current_int_block[:,:-1]))

    itint = Etemp
    return itint  
```


(sec_complie)=
### Computation of Lie Derivatives

Similarly to the stacked list of iterated integrals,
the following algorithm provides the list of the stacked Lie derivatives:

:::{prf:algorithm} 
:label: matrix_lie_deriv

**Inputs** Given the truncation length $N$, the inputs $u$ of the system in [](#nonlinsys)   

**Output** The matrix $\mathcal{G}$ of the stacking of Lie derivatives $L_{\eta}h(x) $ for $|\eta|\leq N$

$G_0 \leftarrow \textbf{1}_{m} \otimes h$

$G_1 \leftarrow \frac{\partial}{\partial x} G_0\cdot g$

$\mathcal{G} \leftarrow G_1$


For $k$ in $\{1,\cdots, N-1\}$ do:
>    1. $V \leftarrow \textbf{1}_{m} \otimes  \frac{\partial}{\partial x}G_k$ 
>    2. $v \leftarrow [I_{m}\otimes \textbf{1}_{N_{G_{k}}}] g$ 
>    3. $M \leftarrow V\cdot v$
>    4. $G_{k+1} \leftarrow M $
>    5. $\mathcal{G} \leftarrow G_{k+1} \oplus_{v} \mathcal{G}$
:::


The Python code of the above algorithm is the following:

```{code} python
:label: my-program
:caption: Lie derivatives
def iter_lie(h,vector_field,z,Ntrunc):
    import numpy as np
    import sympy as sp
    
    # The number of vector fields is obtained.
    # num_vfield = m
    num_vfield = np.size(vector_field,1)
    
    # total_lderiv = num_input + num_input**2 + ... + num_input**Ntrunc
    total_lderiv = num_vfield*(1-pow(num_vfield, Ntrunc))/(1-num_vfield)
    total_lderiv = int(total_lderiv)
    
    # The list that will contain all the Lie derivatives is initiated. 
    Ltemp = sp.Matrix(np.zeros((total_lderiv, 1), dtype='object'))
    ctrLtemp = np.zeros((Ntrunc+1,1), dtype = 'int')
    
    # ctrLtemp[k] = num_input + num_input**2 + ... + num_input**k,  1<=k<=Ntrunc
    for i in range(Ntrunc):
        ctrLtemp[i+1] = ctrLtemp[i] + num_vfield**(i+1)
    
    
    # The Lie derivative L_eta h(z) of words eta of length 1 are computed 
    LT = sp.Matrix([h]).jacobian(z)*vector_field

    # Transforms the lie derivative from a row vector to a column vector
    LT = LT.reshape(LT.shape[0]*LT.shape[1], 1)
    
    # Adds the computed Lie derivatives to a repository
    Ltemp[:num_vfield, 0] = LT

    for i in range(1, Ntrunc):
        # start_prev_block = num_input + num_input**2 + ... + num_input**(i-1)
        start_prev_block = int(ctrLtemp[i-1])
        # end_prev_block = num_input + num_input**2 + ... + num_input**i
        end_prev_block = int(ctrLtemp[i])
        # end_current_block = num_input + num_input**2 + ... + num_input**(i+1)
        end_current_block = int(ctrLtemp[i+1])
        # num_prev_block = num_input**i
        num_prev_block = end_prev_block - start_prev_block
        # num_current_block = num_input**(i+1)
        num_current_block = end_current_block - end_prev_block

        LT = Ltemp[start_prev_block:end_prev_block,0]
        
        LT = LT.jacobian(z)*vector_field
        # Transforms the lie derivative from a row vector to a column vector
        
        LT = LT.reshape(LT.shape[0]*LT.shape[1], 1)
        # Adds the computed Lie derivatives to the repository
        Ltemp[end_prev_block:end_current_block,:]=LT

    return Ltemp 
```


(sec_numCFS)=
### Numerical Computation of Chen-Fliess Series

From [](#sec_compiterint) and [](#sec_complie),
we have the following equivalent representation of the 
CFS truncated word length $N$:

:::{prf:theorem} 
:label: my-theorem

From the previous section, we have

```{math}
F_c^N[u](t) = \mathcal{U}\cdot \mathcal{G}
```

:::

(sec_CFSpy)=
### CFSpy Package

(sec_simulations)=
## Simulations

The following block of code uses CFSpy to compute the CFS
of a Lotka Volterra system.

```{code} python
:label: my-program
:caption: CFS Computation
from CFS import iter_int, iter_lie, single_iter_int, single_iter_lie

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp

# Define the Lotka-Volterra system
def system(t, x, u1_func, u2_func):
    x1, x2 = x
    u1 = u1_func(t)
    u2 = u2_func(t)
    dx1 = -x1*x2 +  x1 * u1
    dx2 = x1*x2 - x2* u2
    return [dx1, dx2]

# Input 1
def u1_func(t):
    return np.sin(t)

# Input 2
def u2_func(t):
    return np.cos(t)

# Initial condition
x0 = [1/3,2/3]

# Time range
t0 = 0
tf = 3
dt = 0.001
t_span = (t0, tf)

# Simulation of the system
solution = solve_ivp(system, t_span, x0, args=(u1_func, u2_func), dense_output=True)

# Partition of the time interval
t = np.linspace(t_span[0], t_span[1], int((tf-t0)//dt+1))
y = solution.sol(t)

# Define the symbolic variables
x1, x2 = sp.symbols('x1 x2')
x = sp.Matrix([x1, x2])


# Define the system symbolically
g = sp.transpose(sp.Matrix([[-x1*x2, x1*x2], [x1, 0], [0, - x2]]))

# Define the output symbolically
h = x1

# The truncation of the length of the words that index the Chen-Fliess series
Ntrunc = 4

# Coefficients of the Chen-Fliess series evaluated at the initial state
Ceta = np.array(iter_lie(h,g,x,Ntrunc).subs([(x[0], 1/3),(x[1], 2/3)]))

# inputs as arrays
u1 = np.sin(t)
u2 = np.cos(t)

# input array
u = np.vstack([u1, u2])

# List of iterated integral
Eu = iter_int(u,t0, tf, dt, Ntrunc)

# Chen-Fliess series
F_cu = x0[0]+np.sum(Ceta*Eu, axis = 0)

# Graph of the output and the Chen-Fliess series
plt.figure(figsize = (12,5))
plt.plot(t, y[0].T)
plt.plot(t, F_cu, color='red', linewidth=5, linestyle = '--', alpha = 0.5)
plt.xlabel('$t$')
plt.ylabel('$x_1$')
plt.legend(['Output of the system','Chen-Fliess series'])
plt.grid()
plt.show()
```


%(sec_futurework)=
%## Future Work
%Accuracy, stability and faster computation

(sec_conclusions)=
## Conclusions

This work has presented a numerical way to compute the Chen-Fliess series in Python package.
For this the definition of the iterative integral has being equivalently modified similar
to the Lie derivative. The algorithms are provided. As future work, this need to be
optimized for a faster computation.
