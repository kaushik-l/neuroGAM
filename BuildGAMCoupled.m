function models = BuildGAMCoupled(xt,yt,Yt,prs)

% BUILDGAMCOUPLED: Fit generalised additive models to neural data.
%   Model: g(r) = sum_i[f_i(x_i)], where r denotes the mean response, x_i 
%   is the i-th input variable and f_i(x_i) is the tuning of y to x_i
%   (fit to data), and g is the link function denoting neuronal 
%   nonlinearity. Several variants of the model are fit, where each variant
%   corresponds to a unique combination of the input variables x_i. The
%   variant with the highest likelihood is chosen via forward search.
%
% INPUTS:
% xt is an 1 x N cell array where the i-th cell is an array containing 
% values of the i-th variable x_i. Each array is either T x 1 or T x 2 
% depending on whether x_i is 1-dimensional (e.g. speed) or 2-dimensional 
% (e.g. position), where T is the total number of observations.
%
% yt is a T x 1 array of spike counts
%
% prs is a structure specifying analysis parameters. The contents of the
% structure are as follows and must be entered in this order.
% prs.varname   : 1 x N cell array of names of the input variables.
%                 only used for labeling plots
% prs.vartype   : 1 x N cell array of types ('1D','1Dcirc' or '2D') of the input variables. 
%                 used for applying smoothness penalty on tuning functions
% prs.nbins     : 1 x N cell array of number of bins to discretise input variables. 
%                 determines the resolution of the tuning curves
% prs.binrange  : 1 x N cell array of 2 x 1 vectors specifying lower and upper bounds of input variables. 
%                 used to determine bin edges
% prs.nfolds    : Number of folds for cross-validation.
% prs.dt        : Time between consecutive observation samples. 
%                 used for converting weights f_i to firing rate
% prs.filtwidth : Width of gaussian filter (in samples) to smooth spike train. 
%                 only used for computing % variance explained
% prs.linkfunc  : Name of the link function g ('log','identity' or 'logit').
% prs.lambda    : 1 x N cell array of hyper-parameters for imposing smoothness prior on tuning functions. 
%                 use 0 to impose no prior
% prs.alpha     : Significance level for comparing likelihood values. 
%                 used for model selection
%
% OUTPUT:
% models is a structure containing the results of fitting different model
% variants. It contains the following fields.
% models.class              : M x 1 cell array containing the class of each of the M model variants.
%                             Each cell is a 1 x N logical array that indicates which of the N input
%                             variables were included in the that model variant
% models.testFit            : M x 1 cell array containing results of testing each model variant on the test sets.
%                             Each cell is a k x 6 matrix where the k-th row contains the results of the k-th test set.
%                             The 6 columns contain fraction of variance explained, pearson correlation between actual 
%                             and predicted response, log likelihood, mean-squared prediction error, total number of
%                             spikes, and total number of observations in that order
% models.trainFit           : M x 1 cell array containing results of testing each model variant on the training sets.
% models.wts                : M x 1 cell array of best-fit weights f_i obtained for each of the M model variants.
%                             f_i is empty if x_i was not included in the model variant
% models.x                  : 1 x N cell array of bin centres at which tunings are computed for the input variables.
% models.bestmodel          : The model variant with the highest validated log likelihood.
% models.marginaltunings    : M x 1 cell array of tuning functions for each of the M model variants.
%                             tuning to x_i is empty if x_i was not included in the model variant

%% Run the uncoupled GAM model to determine components of xt that drive response
Uncoupledmodel = BuildGAM(xt,yt,prs);
bestUncoupledmodel = Uncoupledmodel.bestmodel;
if ~isnan(bestUncoupledmodel), bestinputs = Uncoupledmodel.class{Uncoupledmodel.bestmodel};
else, bestinputs = true(1,size(xt,2)); end
xt = xt(:,bestinputs); % use only the best inputs

%% number of input variables
nvars = length(xt);

%% load analysis parameters
prs = struct2cell(prs);
[~,xtype,nbins, binrange,nfolds,dt,filtwidth,linkfunc,lambda,alpha,~] = deal(prs{:});

%% define undefined analysis parameters
if isempty(alpha), alpha = 0.05; end
if isempty(lambda), lambda = cell(1,nvars+1); lambda(:) = {5e1}; % if user did not provide hyperparameters
elseif length(lambda) == nvars, lambda(nvars+1) = {5e1}; end % if user did not provide a hyperparameter for coupling
if isempty(linkfunc), linkfunc = 'log'; end
if isempty(filtwidth), filtwidth = 3; end
if isempty(nfolds), nfolds = 10; end
if isempty(nbins)
    nbins = cell(1,nvars); nbins(:) = {10}; % default: 10 bins
    nbins(strcmp(xtype,'2D')) = {[10,10]};
end
if isempty(binrange)
    binrange = mat2cell([min(cell2mat(xt));max(cell2mat(xt))],2,strcmp(xtype,'2D')+1);
    binrange(strcmp(xtype,'event')) = {[-0.36;0.36]}; % default: -360ms to 360ms temporal kernel
end
if isempty(dt), dt = 1; end
% express bin range in units of dt for temporal kernels
indx = find(strcmp(xtype,'event'));
for i=indx, binrange{i} = round(binrange{i}/dt); end

%% select only the best inputs
xtype = xtype(bestinputs);
nbins = nbins(bestinputs);
binrange = binrange(bestinputs);
lambdaY = lambda(end);
lambda = lambda(bestinputs);


%% compute inverse-link function
if strcmp(linkfunc,'log')
    invlinkfunc = @(x) exp(x);
elseif strcmp(linkfunc,'identity')
    invlinkfunc = @(x) x;
elseif strcmp(linkfunc,'logit')
    invlinkfunc = @(x) exp(x)./(1 + exp(x));
end

%% encode variables in 1-hot format
x = cell(1,nvars); % 1-hot representation of xt
xc = cell(1,nvars); % bin centres
nprs = cell(1,nvars); % number of parameters (weights)
for i=1:nvars
    [x{i},xc{i},nprs{i}] = Encode1hot(xt{i}, xtype{i}, binrange{i}, nbins{i});
    Px{i} = sum(x{i})/size(x{i},1); 
    if strcmp(xtype{i},'event'), xc{i} = xc{i}*dt; end
end

%% binarise spikes and combine them with input variables
Y = double(Yt>0); % (bins with >1 spike counted as 1 spike ---> if this matters, use small enough bins)
Ytype = {'event'}; % spikes are events
nprsY = {size(Y,2)};

%% combine input variables with spikes from other neurons
X = [cell2mat(x) Y];
Px = [Px sum(Y)/size(Y,1)];
Xtype = [xtype Ytype];
nprs = cell2mat([nprs nprsY]);
Lambda = [lambda lambdaY];
nvars = nvars + 1;

%% define filter to smooth the firing rate
t = linspace(-2*filtwidth,2*filtwidth,4*filtwidth + 1);
h = exp(-t.^2/(2*filtwidth^2));
h = h/sum(h);

%% fit coupled models
fprintf(['...... Fitting fully coupled model with ' linkfunc '-link\n']);
[Coupledmodel.testFit,Coupledmodel.trainFit,Coupledmodel.wts] = FitModel(X,Xtype,nprs,yt,dt,h,nfolds,Lambda,linkfunc,invlinkfunc);
Coupledmodel.x = [xc 1:size(Yt,2)];

%% match weights 'wts' to corresponding inputs 'x'
Coupledmodel.wts = mat2cell(Coupledmodel.wts,1,nprs);

%% compare uncoupled model vs coupled model
if ~isnan(bestUncoupledmodel), UncoupledtestFit = Uncoupledmodel.testFit{bestUncoupledmodel};
else, UncoupledtestFit = Uncoupledmodel.testFit{end}; end
CoupledtestFit = Coupledmodel.testFit;
UncoupledLLvals = UncoupledtestFit(:,3); % 3rd column contains likelihood values
CoupledLLvals = CoupledtestFit(:,3);
[pval1,~] = signrank(CoupledLLvals,UncoupledLLvals,'tail','right');
[pval2,~] = signrank(CoupledLLvals,UncoupledLLvals,'tail','left');
if (pval1<alpha || pval2<alpha), models.LLRcoupling = mean(CoupledLLvals) - mean(UncoupledLLvals);
else, models.LLRcoupling = 0; end

%% convert weights to response rate (tuning curves) & wrap 2D tunings if any
for j=1:nvars
    other_factors = sum(cellfun(@(x,y) sum(x.*y), Coupledmodel.wts(1:nvars ~= j), Px(1:nvars ~= j)));
    Coupledmodel.marginaltunings{j} = invlinkfunc(Coupledmodel.wts{j} + other_factors)/dt;
    if strcmp(Xtype{j},'2D'), Coupledmodel.marginaltunings{j} = reshape(Coupledmodel.marginaltunings{j},nprs(j)); end
end

%% output
models.Coupledmodel = Coupledmodel;
models.Uncoupledmodel = Uncoupledmodel;