function bestmodel = BackwardEliminate(Model,LLvals,alpha)

%% Description
% This function will select the best model variant using forward search. If
% likelihoods of two models are indistinguishable, the one with fewer
% parameter is selected.

%%
Model = cell2mat(Model)';
for i=flip(unique(sum(Model)))
    if i==max(sum(Model)) % select the all-variable model
        bestmodel = find(sum(Model) == max(sum(Model)));
    else % select the best model from among those containing i-1 variables
        indx1 = (sum(Model)==i); % all models containing i variables
        indx2 = (sum(Model(Model(:,bestmodel)>0,:),1)==i);  % all models containing variables in the current best model (with i+1 variables)
        bestcandidate = (nanmean(LLvals) == max(nanmean(LLvals(:,indx1 & indx2))));
        % significance test :: best new candidate vs current best model
        [pval,~] = signrank(LLvals(:,bestcandidate),LLvals(:,bestmodel),'tail','left');
        if pval<alpha, break;
        else, bestmodel = find(bestcandidate); end
    end     
end
if (isempty(bestmodel) || signrank(LLvals(:,bestmodel),0,'tail','right') > alpha), bestmodel = nan; end % best model better than null model?