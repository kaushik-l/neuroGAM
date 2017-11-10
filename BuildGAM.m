function models = BuildGAM(xt,xtype,yt,dt,binrange,nbins,nfolds,filtwidth,modelname,lambda,alpha)

%% Description
% The model: r = f(W*theta), where r is the predicted # of spikes, W is a
% matrix of one-hot vectors describing variable values, theta is the 
% vector of parameters (fit to data), and f is the neural nonlinearity.

%% check dimensions of xt
nt = size(xt,1);
nvars= size(xt,2);
if nvars > nt
    warning('xt has more columns than rows... applying transpose');
    xt = xt';
    nt = size(xt,1);
    nvars = size(xt,2);
end

%% define undefined analysis parameters
if nargin<10, alpha = 0.05; end
if nargin<9, modelname = 'LNP'; end
if nargin<8, filtwidth = 3; end
if nargin<7, nfolds = 10; end
if nargin<6, nbins = cell(1,nvars); nbins(:) = {10}; end
if nargin<5, binrange = []; end
if nargin<4, dt = 1; end

%% define bin range
if isempty(binrange), binrange = mat2cell([max(xt);min(xt)],2,ones(1,nvars)); end

%% encode variables in 1-hot format
x = cell(1,nvars); % 1-hot representation of xt
xc = cell(1,nvars); % bin centres
nprs = cell(1,nvars); % number of parameters (weights)
for i=1:nvars
    [x{i},xc{i},nprs{i}] = Encode1hot(xt(:,i), binrange{i}, nbins{i});
end

%% define model combinations to fit
nModels = sum(arrayfun(@(k) nchoosek(nvars,k), 1:nvars));
X = cell(nModels,1);
Xtype = cell(nModels,1);
Nprs = cell(nModels,1);
Lambda = cell(nModels,1);
ModelCombo = arrayfun(@(k) mat2cell(nchoosek(1:nvars,k),ones(nchoosek(nvars,k),1)),1:nvars,'UniformOutput',false);
ModelCombo = vertcat(ModelCombo{:});
Model = cell(nModels,1); for i=1:nModels, Model{i} = false(1,nvars); end
for i=1:nModels
    Model{i}(ModelCombo{i})=true;
    X{i} = cell2mat(x(Model{i})); % X{i} stores inputs for the i^th model
    Xtype{i} = xtype(Model{i});
    Nprs{i} = cell2mat(nprs(Model{i}));
    Lambda{i} = lambda(Model{i});
end

%% define filter to smooth the firing rate
t = linspace(-2*filtwidth,2*filtwidth,4*filtwidth + 1);
h = exp(-t.^2/(2*filtwidth^2));
h = h/sum(h);

%% fit all models
fprintf('......(1/2) Fitting generalized additive models\n');
models.class = Model; models.testFit = cell(nModels,1); models.trainFit = cell(nModels,1); models.wts = cell(nModels,1);
for n = 1:nModels
    fprintf('\t- Fitting model %d of %d\n', n, nModels);
    if strcmp(modelname,'LNP')
        [models.testFit{n},models.trainFit{n},models.wts{n}] = FitModel(X{n},Xtype{n},Nprs{n},yt,dt,h,nfolds,Lambda{n});
    end
end
models.x = xc;

%% select best model
fprintf('......(2/2) Performing forward model selection\n');
testFit = cell2mat(models.testFit);
nrows = size(testFit,1);
LLvals = reshape(testFit(:,3),nfolds,nrows/nfolds); % 3rd column contains likelihood values
models.bestmodel = ForwardSelect(Model,LLvals,alpha);

%% match weights 'wts' to corresponding inputs 'x'
models.wts = cellfun(@(x,y) mat2cell(x,1,cell2mat(nbins).*y),models.wts,models.class,'UniformOutput',false);