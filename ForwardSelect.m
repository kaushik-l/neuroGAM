function bestmodel = ForwardSelect(Model,LLvals,alpha)

Model = cell2mat(Model)';
nvars = size(Model,1);
for i=1:nvars
    if i==1 % select the best 1-variable model
        bestmodel = find(nanmean(LLvals) == max(nanmean(LLvals(:,sum(Model)==i))));
    else % select the best i-variables model from among those containing the best i-1 variables
        indx1 = (sum(Model)==i); % all models containing i variables
        indx2 = (sum(Model(Model(:,bestmodel)>0,:)',2)==i-1)';  % all models containing best i-1 variables
        bestcandidate = (nanmean(LLvals) == max(nanmean(LLvals(:,indx1 & indx2))));
        % significance test :: best new candidate vs current best model
        [pval,~] = signrank(LLvals(:,bestcandidate),LLvals(:,bestmodel),'tail','right');
        if pval>alpha, break;
        else, bestmodel = bestcandidate; end
    end     
end

bestmodel = find(bestmodel);
if signrank(LLvals(:,bestmodel),0,'tail','right') > alpha, bestmodel = nan; end % best model better than null model?