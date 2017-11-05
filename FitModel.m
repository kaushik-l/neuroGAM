function [testFit,trainFit,param_mean] = FitModel(X,Xtype,Nprs,y,dt,h,nfolds,Lambda)

%% Description
% This function will section the data into nfolds different portions. Each 
% portion is drawn from chunks across the entire recording session. The 
% chunking is done to ensure that each portion includes data from different 
% time points in the experiment (rather than restricted to beginning or the
% end). It will then fit the model to nfolds-1 sections, and test the model 
% performance on the remaining section. This procedure will be repeated 
% nfolds times, with all possible unique testing sections. The fraction of 
% variance explained, the mean-squared error, the log-likelihood increase, 
% and the mean square error will be computed for each test data set. In 
% addition, the learned parameters will be saved for each section of the data.

%% Section the data for k-fold cross-validation and initialise output
nchunks = 5; % divide data into these many chunks (5 is large enough for experiments that last <3 hrs)
[~,nprs] = size(X); % number of parameters to fit
nsections = nfolds*nchunks;

% divide the data up into nfolds*nchunks pieces
edges = round(linspace(1,numel(y)+1,nsections+1));

% initialize outputs
testFit = nan(nfolds,6); % to hold 6 values: var ex, correlation, llh increase, mse, # of spikes, length of test data
trainFit = nan(nfolds,6); % var ex, correlation, llh increase, mse, # of spikes, length of train data
paramMat = nan(nfolds,nprs);

%% perform k-fold cross validation
for k = 1:nfolds
    fprintf('\t\t- Cross validation fold %d of %d\n', k, nfolds);
    
    % get test data for the kth fold - comes from chunks across entire session
    test_ind = cell2mat(arrayfun(@(j) edges((j-1)*nfolds + k):edges((j-1)*nfolds + k + 1)-1, 1:nchunks,'UniformOutput',false));
    
    test_spikes = y(test_ind); %test spiking
    smooth_spikes_test = conv(test_spikes,h,'same'); %returns vector same size as original
    smooth_fr_test = smooth_spikes_test./dt;
    test_X = X(test_ind,:);
    
    % get training data
    train_ind = setdiff(1:numel(y),test_ind);
    train_spikes = y(train_ind);
    smooth_spikes_train = conv(train_spikes,h,'same'); %returns vector same size as original
    smooth_fr_train = smooth_spikes_train./dt;
    train_X = X(train_ind,:);
    
    % train the model
    opts = optimset('Gradobj','on','Hessian','on','Display','off');    
    data{1} = train_X; data{2} = train_spikes;
    if k == 1, init_param = 1e-3*randn(nprs, 1); % initialise random parameters for the first training set
    else, init_param = param; end % use final parameters from previous training set
    
    param = fminunc(@(param) ln_poisson_model(param,data,Xtype,Nprs,Lambda),init_param,opts); % fit parameters of LNP model
    
    %% %%%%%%%%%%% TEST DATA %%%%%%%%%%%%%
    % compute model predicted firing rate
    fr_hat_test = exp(test_X * param)/dt;
    smooth_fr_hat_test = conv(fr_hat_test,h,'same'); %returns vector same size as original
    
    % variance explained
    sse = sum((smooth_fr_hat_test-smooth_fr_test).^2);
    sst = sum((smooth_fr_test-mean(smooth_fr_test)).^2);
    varExplain_test = 1-(sse/sst);
    
    % linear correlation
    correlation_test = corr(smooth_fr_test,smooth_fr_hat_test,'type','Pearson');
    
    % log-likelihood increase from "mean firing rate model" - NO SMOOTHING
    r = exp(test_X * param); n = test_spikes; meanFR_test = nanmean(test_spikes);     
    log_llh_test_model = nansum(r-n.*log(r)+log(factorial(n)))/sum(n); %note: log(gamma(n+1)) will be unstable if n is large (which it isn't here)
    log_llh_test_mean = nansum(meanFR_test-n.*log(meanFR_test)+log(factorial(n)))/sum(n);
    log_llh_test = (-log_llh_test_model + log_llh_test_mean); % nats/spike
    log_llh_test = log_llh_test/log(2); % convert to bits/spike
    
    % mean-squared-error
    mse_test = nanmean((smooth_fr_hat_test-smooth_fr_test).^2);
    
    % fill in all the relevant values for the test data from the kth fold
    testFit(k,:) = [varExplain_test correlation_test log_llh_test mse_test sum(n) numel(test_ind)];
    
    %% %%%%%%%%%%% TRAINING DATA %%%%%%%%%%%
    % compute the firing rate
    fr_hat_train = exp(train_X * param)/dt;
    smooth_fr_hat_train = conv(fr_hat_train,h,'same'); %returns vector same size as original
    
    % variance explained
    sse = sum((smooth_fr_hat_train-smooth_fr_train).^2);
    sst = sum((smooth_fr_train-mean(smooth_fr_train)).^2);
    varExplain_train = 1-(sse/sst);
    
    % correlation
    correlation_train = corr(smooth_fr_train,smooth_fr_hat_train,'type','Pearson');
    
    % log-likelihood
    r_train = exp(train_X * param); n_train = train_spikes; meanFR_train = nanmean(train_spikes);   
    log_llh_train_model = nansum(r_train-n_train.*log(r_train)+log(gamma(n_train+1)))/sum(n_train);
    log_llh_train_mean = nansum(meanFR_train-n_train.*log(meanFR_train)+log(gamma(n_train+1)))/sum(n_train);
    log_llh_train = (-log_llh_train_model + log_llh_train_mean);
    log_llh_train = log_llh_train/log(2); % convert to bits/spike
    
    % mean-squared-error
    mse_train = nanmean((smooth_fr_hat_train-smooth_fr_train).^2);
    
    % fill in all the relevant values for the training data from the kth fold
    trainFit(k,:) = [varExplain_train correlation_train log_llh_train mse_train sum(n_train) numel(train_ind)];

    % save the parameters
    paramMat(k,:) = param;
end

param_mean = nanmean(paramMat);
