---
title: Echo State Networks for Text Classification
---

# Model

An *echo state network (ESN)* is a recurrent neural network composed of
$N_u$ input units, $N_x$ hidden units composing the so-called
*reservoir*, and $N_y$ output units. The input units project onto the
reservoir $(\mathbf{W_{in}})$, which is itself recurrently connected
$(\mathbf{W_{res}})$, and projects onto the output units
$(\mathbf{W_{out}})$.

![An echo state network (ESN).](./figures/esn.png)
<img src="./figures/esn.png" width="48">

Here, we consider *Leaky Integrator ESNs*. The inputs, reservoir state
and outputs of the network at time $t > 0$ are denoted by
$\mathbf{u}(t) \in \mathbb{R}^{N_u}$,
$\mathbf{x}(t) \in \mathbb{R}^{N_x}$ and
$\mathbf{y}(t) \in \mathbb{R}^{N_y}$, respectively. The state
$\mathbf{x}(0)$ is the *initial state*. The dynamics of the network is
then given by the following equations:

\begin{eqnarray*}
\mathbf{\tilde{x}}(t+1) & = & f_{res} \left( \mathbf{W_{in}} [\mathbf{1}, \mathbf{u}(t+1)] + \mathbf{W_{res}} \mathbf{x}(t) \right) \\
\mathbf{x}(t+1)         & = & (1-\alpha) \mathbf{x}(t) + \alpha \mathbf{\tilde{x}}(t+1) \\
\mathbf{y}(t+1)         & = & f_{out} \left( \mathbf{W_{out}} [\mathbf{1}, \mathbf{x}(t+1)] \right)
\end{eqnarray*}

where $[\mathbf{a}, \mathbf{b}]$ denotes the concatenation of
$\mathbf{a}$ and $\mathbf{b}$, $\mathbf{x}(0)$ is the *initial state*,
$f_{res}$ and $f_{out}$ are the *activation functions* of the reservoir
and output cells (applied component-wise), and $\alpha$ is the *leaking
rate* ($0 \leq \alpha \leq 1$).

The leaking rate controls the update speed of the reservoir dynamics.
The input weights $\mathbf{W_{in}}$ are initialized randomly from a
uniform distribution $\mathcal{U}(-a,a)$, where $a$ is the *input
scaling*, and kept fixed during the whole training process. The input
scaling determines the extent of nonlinearity of the reservoir response.
The reservoir weights $\mathbf{W_{res}}$ are drawn from the uniform or
Gaussian distribution, then randomly set to $0$ with a given *sparsity
rate*, and finally rescaled in order to have a specific *spectral
radius* $\rho$ (usually, we choose $\rho < 1$). The reservoir weights
$\mathbf{W_{res}}$ are also kept fixed during training. Only the output
$\mathbf{W_{out}}$ are trainable.

# Training

In an ESN, only the output weights $\mathbf{W_{out}}$ are trained. The
training process can be described as follows. Consider some training set
$\mathcal{S}$ composed of temporal inputs and associated targets, i.e.,

\begin{eqnarray*}
\mathcal{S} & = & \left\{ \left( \mathbf{u}(t), \mathbf{y^{target}}(t) \right) : t = 1, \dots, T \right\}.
\end{eqnarray*}

Let $\mathbf{x}(1), \dots ,\mathbf{x}(T)$ and
$\mathbf{y}(1), \dots ,\mathbf{y}(T)$ be the successive reservoir states
and predictions obtained when running the ENS on inputs
$\mathbf{u}(1), \dots ,\mathbf{u}(T)$, respectively. Then, the output
weights $\mathbf{W_{out}}$ are computed by minimizing some cost function
$\mathcal{L}$ of the predictions and targets via any desired learning
algorithm -- e.g., a simple Ridge regression. Usually, some initial
transient of the ESN dynamics is used as a warm-up of the reservoir, and
$\mathbf{W_{out}}$ is computed on the basis of the remaining suffix of
collected states, predictions and and targets.

Classical temporal tasks involve time series where each point is
associated with a corresponding target. By contrast, in the present
case, the task comprises *multiple* time series as inputs -- the
successive embedded texts -- each of which being associated with only
*one* output target -- its corresponding class. We propose a customized
training process targeted at this *many-to-one* paradigm.

Our training paradigm consist of the four following steps:

:   1.  Embed the successive texts (FastTest, GloVe, etc.);
    2.  Pass the embedded texts into the ESN;
    3.  Merge the reservoir states associated to the successive texts
        (last, mean, etc.);
    4.  Learn the association between the merged states and associated
        targets.

The training process is illustrated in the figure below.

![Customized training paradigm of an echo state network.](./figures/training.png)
