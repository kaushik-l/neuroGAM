# neuroGAM
fit generalised additive models to neural data

## Objective
The generalised additive model can be written as:

<a href="http://www.codecogs.com/eqnedit.php?latex=g(E(\mathbf{y}))=\sum_{i=1}^{N}f_i(\mathbf{x_i})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?g(E(\mathbf{y}))=\sum_{i=1}^{N}f_i(\mathbf{x_i})" title="g(E(\mathbf{y}))=\sum_{i=1}^{N}f_i(\mathbf{x_i})" /></a>

where <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x_i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x_i}" title="\mathbf{x_i}" /></a> is the <a href="http://www.codecogs.com/eqnedit.php?latex=i^{th}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?i^{th}" title="i^{th}" /></a> input variable
, <a href="http://www.codecogs.com/eqnedit.php?latex=f_i(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_i(.)" title="f_i(.)" /></a> is any generic (nonlinear) function operating on <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x_i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x_i}" title="\mathbf{x_i}" /></a>
, <a href="http://www.codecogs.com/eqnedit.php?latex=g(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?g(.)" title="g(.)" /></a> is the link function
, <a href="http://www.codecogs.com/eqnedit.php?latex=E(\mathbf{y})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?E(\mathbf{y})" title="E(\mathbf{y})" /></a> is the expectation value of the response
, and <a href="http://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="http://latex.codecogs.com/gif.latex?N" title="N" /></a> is the total number of input variables.

Given response <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{y}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{y}" title="\mathbf{y}" /></a> and inputs <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x_i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x_i}" title="\mathbf{x_i}" /></a>, the goal is to recover <a href="http://www.codecogs.com/eqnedit.php?latex=f_i(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_i(.)" title="f_i(.)" /></a> under some assumed link function <a href="http://www.codecogs.com/eqnedit.php?latex=g(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?g(.)" title="g(.)" /></a>. This can be solved by computing the maximum a posteriori (MAP) estimate <a href="http://www.codecogs.com/eqnedit.php?latex=\widehat{f_i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\widehat{f_i}" title="\widehat{f_i}" /></a> as:

<a href="http://www.codecogs.com/eqnedit.php?latex=\widehat{f_i}=\begin{matrix}&space;\mathrm{argmax}\\&space;f_i\\&space;\end{matrix}&space;L(f_i|\mathbf{y},\mathbf{x_i})P(f_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\widehat{f_i}=\begin{matrix}&space;\mathrm{argmax}\\&space;f_i\\&space;\end{matrix}&space;L(f_i|\mathbf{y},\mathbf{x_i})P(f_i)" title="\widehat{f_i}=\begin{matrix} \mathrm{argmax}\\ f_i\\ \end{matrix} L(f_i|\mathbf{y},\mathbf{x_i})P(f_i)" /></a>

where <a href="http://www.codecogs.com/eqnedit.php?latex=L(f_i|\mathbf{y},\mathbf{x_i})=P(\mathbf{y}|\mathbf{x_i};&space;f_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L(f_i|\mathbf{y},\mathbf{x_i})=P(\mathbf{y}|\mathbf{x_i};&space;f_i)" title="L(f_i|\mathbf{y},\mathbf{x_i})=P(\mathbf{y}|\mathbf{x_i}; f_i)" /></a> is the model likelihood and <a href="http://www.codecogs.com/eqnedit.php?latex=P(f_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P(f_i)" title="P(f_i)" /></a> is the prior over <a href="http://www.codecogs.com/eqnedit.php?latex=f_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_i" title="f_i" /></a>. For smooth Gaussian priors, we can solve this by maximising:

<a href="http://www.codecogs.com/eqnedit.php?latex=log&space;L(f_i|\mathbf{y,x_i})\&space;-\&space;\lambda&space;\sum_{j=1}^{p_i}\left&space;\|&space;f_{i,j}&space;-f_{i,j&plus;1}\right&space;\|^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?log&space;L(f_i|\mathbf{y,x_i})\&space;-\&space;\lambda&space;\sum_{j=1}^{p_i}\left&space;\|&space;f_{i,j}&space;-f_{i,j&plus;1}\right&space;\|^{2}" title="log L(f_i|\mathbf{y,x_i})\ -\ \lambda \sum_{j=1}^{p_i}\left \| f_{i,j} -f_{i,j+1}\right \|^{2}" /></a>

where <a href="http://www.codecogs.com/eqnedit.php?latex=p_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p_i" title="p_i" /></a> is the total number of parameters used to represent <a href="http://www.codecogs.com/eqnedit.php?latex=f_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_i" title="f_i" /></a> and <a href="http://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> is a hyperparameter capturing its degree of smoothness.

Once we determine the best-fit model parameters <a href="http://www.codecogs.com/eqnedit.php?latex=f_{i,j}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_{i,j}" title="f_{i,j}" /></a> are fit, we can construct marginal tuning functions that describe the tuning of the response to each input variable as:

<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{h_{i}}(\mathbf{x_i})&space;=&space;g^{-1}(\widehat{\mathbf{f_{i}}})\sum_{j=1}^{p_k}&space;\prod_{k=1,k\neq&space;i}^{N}\widehat{f_{k,j}}P(x_{k,j})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{h_{i}}(\mathbf{x_i})&space;=&space;g^{-1}(\widehat{\mathbf{f_{i}}})\sum_{j=1}^{p_k}&space;\prod_{k=1,k\neq&space;i}^{N}\widehat{f_{k,j}}P(x_{k,j})" title="\mathbf{h_{i}}(\mathbf{x_i}) = g^{-1}(\widehat{\mathbf{f_{i}}})\sum_{j=1}^{p_k} \prod_{k=1,k\neq i}^{N}\widehat{f_{k,j}}P(x_{k,j})" /></a>

## Structuring your data

To fit the model using your data, you need to use the function ``BuildGAM``. ``BuildGAM`` takes in three inputs ``xt``, ``yt``, and ``prs``.

``xt`` must an N x 1 cell array of input variables where N is the total number of input variables. Each cell in ``xt`` corresponds to one of the input variables. If <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x_i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x_i}" title="\mathbf{x_i}" /></a> is a one dimensional variable, ``xt{i}`` must be a T x 1 vector, 
T being the total number of observations. If <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x_i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x_i}" title="\mathbf{x_i}" /></a> is two-dimensional e.g. position, then the corresponding ``xt{i}`` must be a T x 2 array, with the two columns corresponding to the two dimensions.

``yt`` must be a T x 1 array of spike counts. It is advisable to record your observations using a sampling rate of at least ``50Hz`` so that ``yt`` is mostly comprised of 0s and 1s.

``prs`` is a structure specifying analysis parameters. The fields of this structure **must be created in the following order**:

``prs.varname``     1 x N cell array of names of the input variables (only used for labeling plots)  
``prs.vartype``     1 x N cell array of types (``'1D'``,``'1Dcirc'`` or ``'2D'``) of the input variables  
``prs.nbins``       1 x N cell array of number of bins to discretise input variables  
``prs.binrange``    1 x N cell array of 2 x 1 vectors specifying lower and upper bounds of the input variables  
``prs.nfolds``      Number of folds for cross-validation  
``prs.dt``          Time (in secs) between consecutive observation samples (1/samplingfrequency)  
``prs.filtwidth``   Width of gaussian filter (in samples) to smooth spike train  
``prs.modelname``   Name of the model (``'LNP'``,``'LP'`` or ``'Logistic'``)  
``prs.lambda``      1 x N cell array of hyper-parameters for imposing smoothness prior on tuning functions  
``prs.alpha``       Significance level for comparing likelihood values  

For more details about the role of these parameters, use ```help BuildGAM``` in MATLAB. Once you have ``xt``, ``yt``, and ``prs``, you can fit the model by running the following command:
```matlab
models = BuildGAM(xt,yt,prs); % the output is saved is the variable called models
```

And then use this command to plot the results:
```matlab
PlotGAM(models,prs);
```

Although meant for neural data, you can use the code in any setting where <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{y}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{y}" title="\mathbf{y}" /></a> is a point process. Checkout ``demo.m`` for examples. [Write](mailto:jklakshm@bcm.edu) to me if you have questions.


## Acknowledgements
This implementation builds on the LNP model described in [Hardcastle et al.](http://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf), implemented [here](https://github.com/GiocomoLab/ln-model-of-mec-neurons) for MEC neurons.
