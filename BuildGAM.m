function models = BuildGAM(xt,yt,prs)

% BUILDGAM: Fit generalised additive models to neural data.
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
% prs.vartype   : 1 x N cell array of types ('1D','1Dcirc','2D' or 'event') of the input variables. 
%                 used for applying smoothness penalty on tuning functions
% prs.nbins     : 1 x N cell array of number of bins to discretise input variables. 
%                 determines the resolution of the tuning curves. If the variable type is 'event', nbins
%                 specifies the number of bins to discretise the temporal kernel corresponding to that event.
% prs.binrange  : 1 x N cell array of 2 x 1 vectors specifying lower and upper bounds of input variables.
%                 used to determine bin edges. If the variable type is 'event', the bounds specify to 
%                 the timerange spanned by the temporal kernel corresponding to that event.
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
% prs.varchoose : 1 x N array of 1s and 0s indicating the inclusion status of each variable. Use 1 to forcibly
%                 include a variable in the bestmodel, 0 to let the method determine when to include a variable
% prs.method    : Method for selecting the best model ('Forward' / 'Backward' / 'FastForward' / 'FastBackward')
%                 'Forward' uses forward-selection, 'Backward' uses backward-elimination
%                 'FastForward' and 'FastBackward' are the corresponding fast implementations (fit a subset rather than all models)
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

%% number of input variables
nvars = length(xt);

%% load analysis parameters
prs = struct2cell(prs);
[xname,xtype,nbins, binrange,nfolds,dt,filtwidth,linkfunc,lambda,alpha,varchoose,method] = deal(prs{:});
method = lower(method);

%% define undefined analysis parameters
if isempty(alpha), alpha = 0.05; end
if isempty(lambda), lambda = cell(1,nvars); lambda(:) = {5e1}; end
if isempty(linkfunc), linkfunc = 'log'; end
if isempty(filtwidth), filtwidth = 3; end
if isempty(nfolds), nfolds = 10; end
if isempty(nbins)
    nbins = cell(1,nvars); nbins(:) = {10}; % default: 10 bins
    nbins(strcmp(xtype,'2D')) = {[10,10]};
end
if isempty(binrange), binrange = []; end
if isempty(dt), dt = 1; end
% define bin range
if isempty(binrange)
    binrange = mat2cell([min(cell2mat(xt));max(cell2mat(xt))],2,strcmp(xtype,'2D')+1);
    binrange(strcmp(xtype,'event')) = {[-0.36;0.36]}; % default: -360ms to 360ms temporal kernel
end
% express bin range in units of dt for temporal kernels
indx = find(strcmp(xtype,'event'));
for i=indx, binrange{i} = round(binrange{i}/dt); end

%% compute inverse-link function
if strcmp(linkfunc,'log')
    invlinkfunc = @(x) exp(x);
elseif strcmp(linkfunc,'identity')
    invlinkfunc = @(x) x;
elseif strcmp(linkfunc,'logit')
    invlinkfunc = @(x) exp(x)./(1 + exp(x));
end
fprintf(['...... Fitting ' linkfunc '-link model\n']);

%% encode variables in 1-hot format
x = cell(1,nvars); % 1-hot representation of xt
xc = cell(1,nvars); % bin centres
nprs = cell(1,nvars); % number of parameters (weights)
for i=1:nvars
    [x{i},xc{i},nprs{i}] = Encode1hot(xt{i}, xtype{i}, binrange{i}, nbins{i});
    Px{i} = sum(x{i}~=0)/size(x{i},1); 
    if strcmp(xtype{i},'event'), xc{i} = xc{i}*dt; end
end

%% define filter to smooth the firing rate
t = linspace(-2*filtwidth,2*filtwidth,4*filtwidth + 1);
h = exp(-t.^2/(2*filtwidth^2));
h = h/sum(h);

%% use appropriate method to fit the model
switch method
    case {'forward','backward'}
        %% define model combinations to fit
        Model = DefineModels(nvars,1:nvars,varchoose);
%         Model = Model(end-nvars:end); % fit only nvars+1 models (full model & models missing one variable)
        %% fit all models
        models = FitModels(Model,x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc);        
        %% select best model
        fprintf('...... Performing model selection\n');
        testFit = cell2mat(models.testFit); nrows = size(testFit,1);
        LLvals = reshape(testFit(:,3),nfolds,nrows/nfolds); % 3rd column contains likelihood values
        if strcmp(method,'forward'), models.bestmodel = ForwardSelect(Model,LLvals,alpha);
        elseif strcmp(method,'backward'), models.bestmodel = BackwardEliminate(Model,LLvals,alpha); end
    case {'fastforward'}
        Model = DefineModels(nvars,1:nvars,varchoose);
        models.class = Model; 
        for n = 1:length(Model), models.testFit{n,1} = nan(nfolds,6); models.trainFit{n,1} = nan(nfolds,6); models.wts{n,1} = nan(1,sum(cell2mat(nprs).*models.class{n})); end
        models = FastForwardSelect(Model,models,x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc,alpha);
    case {'fastbackward'}
        Model = DefineModels(nvars,1:nvars,varchoose);
        models.class = Model; 
        for n = 1:length(Model), models.testFit{n,1} = nan(nfolds,6); models.trainFit{n,1} = nan(nfolds,6); models.wts{n,1} = nan(1,sum(cell2mat(nprs).*models.class{n})); end
        models = FastBackwardEliminate(Model,models,x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc,alpha);
end

%% match weights 'wts' to corresponding inputs 'x'
models.wts = cellfun(@(x,y) mat2cell(x,1,cell2mat(nprs).*y),models.wts,models.class,'UniformOutput',false);
models.x = xc; models.xname = xname; models.xtype = xtype;

%% convert weights to response rate (tuning curves) & wrap 2D tunings if any
for i=1:numel(models.class)
    for j=1:nvars
        if models.class{i}(j) && ~all(isnan(models.wts{i}{j}))
            if isempty(models.wts{i}(j~=1:nvars & models.class{i})) || (strcmp(xtype{j},'event') && all(strcmp(xtype(j~=1:nvars & models.class{i}),'event'))), other_factors = 0; % events don't overlap => need not marginalise
            else, other_factors = sum(cellfun(@(x,y) sum(x.*y), models.wts{i}(j~=1:nvars & models.class{i}), Px(j~=1:nvars & models.class{i}))); end
            models.marginaltunings{i}{j} = invlinkfunc(models.wts{i}{j} + other_factors)/dt;
            if strcmp(xtype{j},'2D'), models.marginaltunings{i}{j} = reshape(models.marginaltunings{i}{j},nbins{j}); end
        else, models.marginaltunings{i}{j} = []; 
        end
    end
end