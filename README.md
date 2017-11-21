# neuroGAM
fit generalised additive models to neural data

## Objective
Generalised additive model (GAM) is a framework to simultaneously estimate the relationship between a response variable and multiple predictor variables. Unlike generalised linear models (GLM), GAMs can account for arbitary nonlinear relationships between predictors and response. Since the relationship between stimulus variables and spikes is often nonlinear and nonmonotonic, GAMs make an attractive option for modeling neural response. The model can be written as:

<a href="https://www.codecogs.com/eqnedit.php?latex=g(E(\mathbf{y}))=\sum_{i=1}^{n}\mathbf{f}_i(\mathbf{x}_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(E(\mathbf{y}))=\sum_{i=1}^{n}\mathbf{f}_i(\mathbf{x}_i)" title="g(E(\mathbf{y}))=\sum_{i=1}^{n}\mathbf{f}_i(\mathbf{x}_i)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a> is the <a href="http://www.codecogs.com/eqnedit.php?latex=i^{th}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?i^{th}" title="i^{th}" /></a> input variable, <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{f}_i(.)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{f}_i(.)" title="\mathbf{f}_i(.)" /></a> is any generic (nonlinear) function operating on <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a>, <a href="http://www.codecogs.com/eqnedit.php?latex=g(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?g(.)" title="g(.)" /></a> is the link function, <a href="http://www.codecogs.com/eqnedit.php?latex=E(\mathbf{y})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?E(\mathbf{y})" title="E(\mathbf{y})" /></a> is the expectation value of the response, and <a href="http://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="http://latex.codecogs.com/gif.latex?n" title="n" /></a> is the total number of input variables.

Given response <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{y}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{y}" title="\mathbf{y}" /></a> and inputs <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a>, the goal is to recover <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{f}_i(.)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{f}_i(.)" title="\mathbf{f}_i(.)" /></a> under some assumed link function <a href="http://www.codecogs.com/eqnedit.php?latex=g(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?g(.)" title="g(.)" /></a>. This can be solved by computing the maximum a posteriori (MAP) estimate <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{\widehat{f}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{\widehat{f}}" title="\mathbf{\widehat{f}}" /></a> as:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{\widehat{f}}=\begin{matrix}&space;\textup{argmax}\\&space;\mathbf{f}&space;\end{matrix}\&space;L(\mathbf{f}|\mathbf{y},\mathbf{x})\&space;P(\mathbf{f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{\widehat{f}}=\begin{matrix}&space;\textup{argmax}\\&space;\mathbf{f}&space;\end{matrix}\&space;L(\mathbf{f}|\mathbf{y},\mathbf{x})\&space;P(\mathbf{f})" title="\mathbf{\widehat{f}}=\begin{matrix} \textup{argmax}\\ \mathbf{f} \end{matrix}\ L(\mathbf{f}|\mathbf{y},\mathbf{x})\ P(\mathbf{f})" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=L(\mathbf{f}|\mathbf{y},\mathbf{x}_i)=P(\mathbf{y}|\mathbf{x};\mathbf{f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\mathbf{f}|\mathbf{y},\mathbf{x})=P(\mathbf{y}|\mathbf{x};\mathbf{f})" title="L(\mathbf{f}|\mathbf{y},\mathbf{x})=P(\mathbf{y}|\mathbf{x};\mathbf{f})" /></a> is the model likelihood and <a href="https://www.codecogs.com/eqnedit.php?latex=P(\mathbf{f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(\mathbf{f})" title="P(\mathbf{f})" /></a> is the prior over <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{f}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{f}" title="\mathbf{f}" /></a>. 

## Algorithm
We begin by discretising the value of each input variable <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a> into <a href="http://www.codecogs.com/eqnedit.php?latex=m_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?m_i" title="m_i" /></a> bins. We recode the value of <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_{i,t}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x}_{i,t}" title="\mathbf{x}_{i,t}" /></a> at each time point <a href="http://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="http://latex.codecogs.com/gif.latex?t" title="t" /></a> as a one-hot binary string <a href="http://www.codecogs.com/eqnedit.php?latex=x_{i1}x_{i2}...x_{im_{i}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{i1}x_{i2}...x_{im_{i}}" title="x_{i1}x_{i2}...x_{im_{i}}" /></a> where the <a href="http://www.codecogs.com/eqnedit.php?latex=j^{\mathrm{th}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?j^{\mathrm{th}}" title="j^{\mathrm{th}}" /></a> bit is:

<a href="http://www.codecogs.com/eqnedit.php?latex=x_{ij}=\left\{\begin{matrix}&space;\quad&space;\&space;\&space;1&space;\&space;\mathrm{if}\&space;\mathbf{x}_i&space;\in&space;j^{\mathrm{th}}&space;\&space;\mathrm{bin}\\&space;0&space;\&space;\mathrm{otherwise}&space;\end{matrix}\right." target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{ij}=\left\{\begin{matrix}&space;\quad&space;\&space;\&space;1&space;\&space;\mathrm{if}\&space;\mathbf{x}_i&space;\in&space;j^{\mathrm{th}}&space;\&space;\mathrm{bin}\\&space;0&space;\&space;\mathrm{otherwise}&space;\end{matrix}\right." title="x_{ij}=\left\{\begin{matrix} \quad \ \ 1 \ \mathrm{if}\ \mathbf{x}_i \in j^{\mathrm{th}} \ \mathrm{bin}\\ 0 \ \mathrm{otherwise} \end{matrix}\right." /></a>


Let <a href="http://www.codecogs.com/eqnedit.php?latex=f_{i,j}=\{\mathbf{f}_i(\mathbf{x}_i)\&space;|\&space;x_{ij}=1\}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_{i,j}=\{\mathbf{f}_i(\mathbf{x}_i)\&space;|\&space;x_{ij}=1\}" title="f_{i,j}=\{\mathbf{f}_i(\mathbf{x}_i)\ |\ x_{ij}=1\}" /></a> denote the value of <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{f}_i(\mathbf{x}_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{f}_i(\mathbf{x}_i)" title="\mathbf{f}_i(\mathbf{x}_i)" /></a> when <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i&space;\in&space;j^{\mathrm{th}}&space;\&space;\mathrm{bin}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x}_i&space;\in&space;j^{\mathrm{th}}&space;\&space;\mathrm{bin}" title="\mathbf{x}_i \in j^{\mathrm{th}} \ \mathrm{bin}" /></a>. If <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{f}_i(\mathbf{x}_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{f}_i(\mathbf{x}_i)" title="\mathbf{f}_i(\mathbf{x}_i)" /></a> is known to vary smoothly, we can write <a href="http://www.codecogs.com/eqnedit.php?latex=P(\mathbf{f})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P(\mathbf{f})" title="P(\mathbf{f})" /></a> as:

<a href="http://www.codecogs.com/eqnedit.php?latex=P(\mathbf{f})=\prod_{i=1}^{n}P(\mathbf{f}_i)=\prod_{i=1}^{n}\prod_{j=1}^{m_i}e^{-\lambda_i(f_{i,j}-f_{i,j&plus;1})^{2}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P(\mathbf{f})=\prod_{i=1}^{n}P(\mathbf{f}_i)=\prod_{i=1}^{n}\prod_{j=1}^{m_i}e^{-\lambda_i(f_{i,j}-f_{i,j&plus;1})^{2}}" title="P(\mathbf{f})=\prod_{i=1}^{n}P(\mathbf{f}_i)=\prod_{i=1}^{n}\prod_{j=1}^{m_i}e^{-\lambda_i(f_{i,j}-f_{i,j+1})^{2}}" /></a>

where we have assumed a factorisable gaussian prior for simplicity, and <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_i" title="\lambda_i" /></a> is a hyperparameter capturing the degree of smoothness of <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{f}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{f}_i" title="\mathbf{f}_i" /></a>. We solve for <a href="http://www.codecogs.com/eqnedit.php?latex=\hat{\mathbf{f}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\hat{\mathbf{f}}" title="\hat{\mathbf{f}}" /></a> by maximising:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathrm{log}\&space;L(\mathbf{f}|\mathbf{y},\mathbf{x})\&space;-&space;\lambda&space;\sum_{i=1}^{n}\sum_{j=1}^{m_i}\left&space;\|&space;f_{i,j}-f_{i,j&plus;1}&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathrm{log}\&space;L(\mathbf{f}|\mathbf{y},\mathbf{x})\&space;-&space;\sum_{i=1}^{n}\lambda_i&space;\sum_{j=1}^{m_i}\left&space;\|&space;f_{i,j}-f_{i,j&plus;1}&space;\right&space;\|^{2}" title="\mathrm{log}\ L(\mathbf{f}|\mathbf{y},\mathbf{x})\ - \lambda \sum_{i=1}^{n}\sum_{j=1}^{m_i}\left \| f_{i,j}-f_{i,j+1} \right \|^{2}" /></a>

Once we determine the best-fit model parameters <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{\hat{f}}_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{\hat{f}}_i" title="\mathbf{\hat{f}}_i" /></a>, we can construct the marginal 'tuning' to each individual input variable <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a> by computing the conditional mean of <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{y}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{y}" title="\mathbf{y}" /></a> given <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a> as:

<a href="http://www.codecogs.com/eqnedit.php?latex=E(\mathbf{y}|\mathbf{x}_i)=g^{-1}\left&space;(\mathbf{\hat{f}}_i&space;\&space;&plus;\&space;\sum_{i=1}^{n}\sum_{k=1,k\neq&space;i}^{m_k}\hat{f}_{k,j}\&space;P(x_{kj}=1)\right&space;)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?E(\mathbf{y}|\mathbf{x}_i)=g^{-1}\left&space;(\mathbf{\hat{f}}_i&space;\&space;&plus;\&space;\sum_{i=1}^{n}\sum_{k=1,k\neq&space;i}^{m_k}\hat{f}_{k,j}\&space;P(x_{kj}=1)\right&space;)" title="E(\mathbf{y}|\mathbf{x}_i)=g^{-1}\left (\mathbf{\hat{f}}_i \ +\ \sum_{i=1}^{n}\sum_{k=1,k\neq i}^{m_k}\hat{f}_{k,j}\ P(x_{kj}=1)\right )" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=P(x_{kj}=1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x_{kj}=1)" title="P(x_{kj})" /></a> denotes the probability that <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_k\in&space;j^{\mathrm{th}}&space;\&space;\mathrm{bin}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x}_k\in&space;j^{\mathrm{th}}&space;\&space;\mathrm{bin}" title="\mathbf{x}_k\in j^{\mathrm{th}} \ \mathrm{bin}" /></a>.

## Structuring your data

To fit the model using your data, you need to use the function ``BuildGAM``. ``BuildGAM`` takes in three inputs ``xt``, ``yt``, and ``prs``.

``xt`` must be an n x 1 cell array containing the values of input variables where n is the total number of input variables. Each cell in ``xt`` corresponds to one input variable. If <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a> is a one dimensional variable, ``xt{i}`` must be a T x 1 vector, 
T being the total number of observations. If <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{x}_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{x}_i" title="\mathbf{x}_i" /></a> is two-dimensional e.g. position, then the corresponding ``xt{i}`` must be a T x 2 array, with the two columns corresponding to the two dimensions.

``yt`` must be a T x 1 array of spike counts. It is advisable to record your observations using a sampling rate of at least ``50Hz`` so that ``yt`` is mostly comprised of 0s and 1s.

``prs`` is a structure specifying analysis parameters. The fields of this structure **must be created in the following order**:

``prs.varname``     1 x n cell array of names of the input variables (only used for labeling plots)  
``prs.vartype``     1 x n cell array of types (``'1D'``,``'1Dcirc'`` or ``'2D'``) of the input variables  
``prs.nbins``       1 x n cell array of number of bins to discretise input variables  
``prs.binrange``    1 x n cell array of 2 x 1 vectors specifying lower and upper bounds of the input variables  
``prs.nfolds``      Number of folds for cross-validation  
``prs.dt``          Time (in secs) between consecutive observation samples (1/samplingfrequency)  
``prs.filtwidth``   Width of gaussian filter (in samples) to smooth spike train  
``prs.linkfunc``    Choice of link function (``'log'``,``'identity'`` or ``'logit'``)  
``prs.lambda``      1 x n cell array of hyper-parameters for imposing smoothness prior on tuning functions  
``prs.alpha``       Significance level for comparing likelihood values  

For more details about the role of these parameters, use ```help BuildGAM``` in MATLAB. Once you have ``xt``, ``yt``, and ``prs``, you can fit the model by running the following command:
```matlab
models = BuildGAM(xt,yt,prs); % the output is saved in the variable called models
```

And then use this command to plot the results:
```matlab
PlotGAM(models,prs); % plot model likelihoods and marginal tuning functions
```

Although meant for neural data, you can use this code to model any point process <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{y}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{y}" title="\mathbf{y}" /></a>. Checkout ``demo.m`` for examples. [Write](mailto:jklakshm@bcm.edu) to me if you have questions.


## Acknowledgements
This implementation builds on the LNP model described in [Hardcastle et al., 2017](http://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf), available [here](https://github.com/GiocomoLab/ln-model-of-mec-neurons) for MEC neurons.
