function models = FastForwardSelect(Model,models,x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc,alpha)

Modelmatrix = cell2mat(Model)'; fit_indx = [];
for i=min(sum(Modelmatrix)):max(sum(Modelmatrix))
    if i==min(sum(Modelmatrix)) % first, select the best of the simplest possible models
        indx = sum(Modelmatrix) == i; fit_indx = [fit_indx find(indx)];
        models_temp = FitModels(Model(indx),x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc);
        models.testFit(indx,1) = models_temp.testFit; models.trainFit(indx,1) = models_temp.trainFit; 
        models.wts(indx,1) = models_temp.wts; models.wtsMat(indx,1) = models_temp.wtsMat;
        models.response(indx,1) = models_temp.response;
        testFit = cell2mat(models.testFit); nrows = size(testFit,1);
        LLvals = reshape(testFit(:,4),nfolds,nrows/nfolds); % 4th column contains likelihood values
        models.bestmodel = find(nanmean(LLvals) == max(nanmean(LLvals(:,sum(Modelmatrix)==i))));
        if isempty(models.bestmodel), break; end
    else % then, select the best i-variables model from among those containing the best i-1 variables
        indx1 = (sum(Modelmatrix)==i); % all models containing i variables
        indx2 = (sum(Modelmatrix(Modelmatrix(:,models.bestmodel)>0,:),1)==i-1);  % all models containing best i-1 variables
        indx = indx1 & indx2; fit_indx = [fit_indx find(indx)];
        models_temp = FitModels(Model(indx),x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc);
        models.testFit(indx,1) = models_temp.testFit; models.trainFit(indx,1) = models_temp.trainFit; 
        models.wts(indx,1) = models_temp.wts; models.wtsMat(indx,1) = models_temp.wtsMat;
        models.response(indx,1) = models_temp.response;
        testFit = cell2mat(models.testFit); nrows = size(testFit,1);
        LLvals = reshape(testFit(:,4),nfolds,nrows/nfolds); % 4th column contains likelihood values
        bestcandidate = (nanmean(LLvals) == max(nanmean(LLvals(:,indx1 & indx2))));
        % significance test :: best new candidate vs current best model
        [pval,~] = signrank(LLvals(:,bestcandidate),LLvals(:,models.bestmodel),'tail','right');
        if pval>alpha, break;
        else, models.bestmodel = find(bestcandidate); end
    end
end
fprintf('...... Performing model selection\n');
if all(isnan(LLvals(:,models.bestmodel))), models.bestmodel = nan;
elseif (isempty(models.bestmodel) || signrank(LLvals(:,models.bestmodel),0,'tail','right') > alpha), models.bestmodel = nan; end % best model better than null model?
if ~isnan(models.bestmodel), bestmodelclass = models.class{models.bestmodel}; end

% only output models that were fit
models.class = models.class(fit_indx);
models.testFit = models.testFit(fit_indx);
models.trainFit = models.trainFit(fit_indx);
models.wts = models.wts(fit_indx);
models.wtsMat = models.wtsMat(fit_indx);
if ~isnan(models.bestmodel), models.bestmodel = find(all(cell2mat(models.class) == bestmodelclass,2)); end