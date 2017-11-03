function models = FitModels(xt,yt,dt,binrange,nbins,nfolds,filtwidth,modelname)

%% Description
% The model: r = f(W*theta), where r is the predicted # of spikes, W is a
% matrix of one-hot vectors describing variable values, theta is the 
% vector of parameters (fit to data), and f is the neural nonlinearity.

%% define undefined analysis parameters
if nargin<7, filtwidth = 3; end
if nargin<6, nfolds = 10; end
if nargin<5, nbins = 10; end
if nargin<4, binrange = []; end

%% check dimensions of xt
nt = size(xt,1);
nvars= size(xt,2);
if nvars > nt
    warning('xt has more columns than rows... applying transpose');
    xt = xt';
    nt = size(xt,1);
    nvars= size(xt,2);
end

%% define bin edges
binedges = zeros(nvars,nbins+1);  % define bin edges for conversion to 1-hot
for i=1:nvars
    if ~isempty(binrange)
        binedges(i,:) = linspace(binrange(1,i),binrange(2,i),nbins+1);
    else
        binedges(i,:) = linspace(min(xt(:,i)),max(xt(:,i)),nbins+1);
    end
end

%% encode variables in 1-hot format
x = zeros(nt,nbins,nvars); % 1-hot representation of xt
xc = zeros(nbins,nvars); % bin centres
for i=1:nvars
    [x(:,:,i),xc(:,i)] = Encode1hot(xt(:,i), binedges(i,:));
end

%% define model combinations to fit
nModels = sum(arrayfun(@(k) nchoosek(nvars,k), 1:nvars));
X = cell(nModels,1);
ModelCombo = arrayfun(@(k) mat2cell(nchoosek(1:nvars,k),ones(nchoosek(nvars,k),1)),1:nvars,'UniformOutput',false);
ModelCombo = vertcat(ModelCombo{:});
Model = cell(nModels,1); for i=1:nModels, Model{i} = false(1,nvars); end
for i=1:nModels
    Model{i}(ModelCombo{i})=true;
    X{i} = reshape(x(:,:,Model{i}),nt,[]); % X{i} stores inputs for the i^th model
end

%% define filter to smooth the firing rate
t = linspace(-2*filtwidth,2*filtwidth,4*filtwidth + 1);
h = exp(-t.^2/(2*filtwidth^2));
h = h/sum(h);

%% fit all models
models.testFit = cell(nModels,1); models.trainFit = cell(nModels,1); models.wts = cell(nModels,1);
for n = 1:nModels
    fprintf('\t- Fitting model %d of %d\n', n, numModels);
    if strcmp(modelname,'LNP')
        [models.testFit{n},models.trainFit{n},models.wts{n}] = FitLNPmodel(X{n},yt,dt,h,Model{n},nfolds);
    end
end
