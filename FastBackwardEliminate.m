function models = FastBackwardEliminate(Model,models,x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc,alpha)

Modelmatrix = cell2mat(Model)'; fit_indx = [];
for i=flip(unique(sum(Modelmatrix)))
    if i==max(sum(Modelmatrix)) % first, select the all-variable model
        indx = sum(Modelmatrix) == i; fit_indx = [fit_indx find(indx)];
        models_temp = FitModels(Model(indx),x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc);
        models.testFit(indx,1) = models_temp.testFit; models.trainFit(indx,1) = models_temp.trainFit; 
        models.wts(indx,1) = models_temp.wts; models.wtsMat(indx,1) = models_temp.wtsMat;
        models.bestmodel = find(sum(Modelmatrix) == max(sum(Modelmatrix)));
        testFit = cell2mat(models.testFit); nrows = size(testFit,1);
        LLvals = reshape(testFit(:,4),nfolds,nrows/nfolds); % 4th column contains likelihood values
    else % then, % select the best model from among those containing i-1 variables
        indx1 = (sum(Modelmatrix)==i); % all models containing i variables
        indx2 = (sum(Modelmatrix(Modelmatrix(:,models.bestmodel)>0,:),1)==i);  % all models containing variables in the current best model (with i+1 variables)
        indx = indx1 & indx2; fit_indx = [fit_indx find(indx)];
        models_temp = FitModels(Model(indx),x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc);
        models.testFit(indx,1) = models_temp.testFit; models.trainFit(indx,1) = models_temp.trainFit; 
        models.wts(indx,1) = models_temp.wts; models.wtsMat(indx,1) = models_temp.wtsMat;
        testFit = cell2mat(models.testFit); nrows = size(testFit,1);
        LLvals = reshape(testFit(:,4),nfolds,nrows/nfolds); % 4th column contains likelihood values
        bestcandidate = (nanmean(LLvals) == max(nanmean(LLvals(:,indx1 & indx2))));
        % significance test :: best new candidate vs current best model
        [pval,~] = signrank(LLvals(:,bestcandidate),LLvals(:,models.bestmodel),'tail','left');
        if pval<alpha, break;
        else, models.bestmodel = find(bestcandidate); end
    end
end
fprintf('...... Performing model selection\n');
if (isempty(models.bestmodel) || signrank(LLvals(:,models.bestmodel),0,'tail','right') > alpha), models.bestmodel = nan; end % best model better than null model?
if ~isnan(models.bestmodel), bestmodelclass = models.class{models.bestmodel}; end

% only output models that were fit
models.class = models.class(fit_indx);
models.testFit = models.testFit(fit_indx);
models.trainFit = models.trainFit(fit_indx);
models.wts = models.wts(fit_indx);
models.wtsMat = models.wtsMat(fit_indx);
if ~isnan(models.bestmodel), models.bestmodel = find(all(cell2mat(models.class) == bestmodelclass,2)); end