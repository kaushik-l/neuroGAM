function bestmodel = ForwardSelect(Model,LLvals,alpha)

%% Description
% This function will select the best model variant using forward search. If
% likelihoods of two models are indistinguishable, the one with fewer
% parameter is selected.

%%
Model = cell2mat(Model)';
nvars = size(Model,1);
for i=1:nvars
    if i==1 % select the best 1-variable model
        bestmodel = find(nanmean(LLvals) == max(nanmean(LLvals(:,sum(Model)==i))));
        if isempty(bestmodel)
            break; 
        end
    else % select the best i-variables model from among those containing the best i-1 variables
        indx1 = (sum(Model)==i); % all models containing i variables
        indx2 = (sum(Model(Model(:,bestmodel)>0,:),1)==i-1);  % all models containing best i-1 variables
        bestcandidate = (nanmean(LLvals) == max(nanmean(LLvals(:,indx1 & indx2))));
        % significance test :: best new candidate vs current best model
        [pval,~] = signrank(LLvals(:,bestcandidate),LLvals(:,bestmodel),'tail','right');
        if pval>alpha, break;
        else, bestmodel = bestcandidate; end
    end     
end

bestmodel = find(bestmodel);
if (isempty(bestmodel) || signrank(LLvals(:,bestmodel),0,'tail','right') > alpha), bestmodel = nan; end % best model better than null model?