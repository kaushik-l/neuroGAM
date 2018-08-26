function models = FastBackwardEliminate(Model,models,X,Xtype,Nprs,yt,dt,h,nfolds,Lambda,linkfunc,invlinkfunc,alpha)

Modelmatrix = cell2mat(Model)';
for i=flip(unique(sum(Modelmatrix)))
    if i==max(sum(Modelmatrix)) % first, select the all-variable model
        indx = sum(Modelmatrix) == i;
        models_temp = FitModels(Model(indx),X(indx),Xtype(indx),Nprs(indx),yt,dt,h,nfolds,Lambda(indx),linkfunc,invlinkfunc);
        models.testFit(indx,1) = models_temp.testFit; models.trainFit(indx,1) = models_temp.trainFit; models.wts(indx,1) = models_temp.wts;
        models.bestmodel = find(sum(Modelmatrix) == max(sum(Modelmatrix)));
    else % then, % select the best model from among those containing i-1 variables
        indx1 = (sum(Modelmatrix)==i); % all models containing i variables
        indx2 = (sum(Modelmatrix(Modelmatrix(:,models.bestmodel)>0,:),1)==i);  % all models containing variables in the current best model (with i+1 variables)
        indx = indx1 & indx2;
        models_temp = FitModels(Model(indx),X(indx),Xtype(indx),Nprs(indx),yt,dt,h,nfolds,Lambda(indx),linkfunc,invlinkfunc);
        models.testFit(indx,1) = models_temp.testFit; models.trainFit(indx,1) = models_temp.trainFit; models.wts(indx,1) = models_temp.wts;
        testFit = cell2mat(models.testFit); nrows = size(testFit,1);
        LLvals = reshape(testFit(:,3),nfolds,nrows/nfolds); % 3rd column contains likelihood values
        bestcandidate = (nanmean(LLvals) == max(nanmean(LLvals(:,indx1 & indx2))));
        % significance test :: best new candidate vs current best model
        [pval,~] = signrank(LLvals(:,bestcandidate),LLvals(:,models.bestmodel),'tail','left');
        if pval<alpha, break;
        else, models.bestmodel = find(bestcandidate); end
    end
end
fprintf('...... Performing model selection\n');
if (isempty(models.bestmodel) || signrank(LLvals(:,models.bestmodel),0,'tail','right') > alpha), models.bestmodel = nan; end % best model better than null model?