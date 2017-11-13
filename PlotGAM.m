function models = PlotGAM(models,prs)

fprintf('...... Plotting results\n');

%% load analysis parameters
prs = struct2cell(prs);
[varname,vartype,~, ~,nfolds,~,~,~,~,~] = deal(prs{:});
nvars = length(varname);

% give each combination of variables a name
nModels = length(models.class);
varlabel = cell(1,nvars); modellabel = cell(1,nModels);
for i=1:nvars
    if strcmp(vartype{i},'2D'), varlabel{i} = varname{i}{1}(1); % use first letter of the variable name to label
    else, varlabel{i} = varname{i}(1); end
end
for i=1:nModels, modellabel{i} = cell2mat(varlabel(models.class{i})); end

%% load model info
testFit = cell2mat(models.testFit);
nrows = size(testFit,1);
bestmodel = models.bestmodel;
LLvals = reshape(testFit(:,3),nfolds,nrows/nfolds); % 3rd column contains likelihood values
Vexp = reshape(testFit(:,1),nfolds,nrows/nfolds); % 1st column contains variance explained
xvals = models.x;
fvals = models.marginaltunings{bestmodel};

%% plot
SS_pix = get(0,'screensize');
figure; set(gcf,'Position',SS_pix); hold on;
Nc = 4; % plot N x 4 panels
Nr = 1 + 1 + ceil(nvars/Nc); % plot log-likelihood , var explained , tuning to each variable

% likelihoods
subplot(Nr,Nc,1:Nc); hold on;
errorbar(nanmean(LLvals),nanstd(LLvals)/sqrt(nfolds),'ok','linewidth',3);
plot(bestmodel,mean(LLvals(:,bestmodel)),'.r','markersize',20);
plot(0.5:nModels+0.5,zeros(nModels+1,1),'--k','linewidth',2);
set(gca,'fontsize',16); box off;
set(gca,'XLim',[0 nModels+1]); set(gca,'XTick',1:nModels);
set(gca,'XTickLabel',modellabel);
legend('Model performance','Selected model','Null model');
ylabel('Log likelihood ratio (bits/spike)','Fontsize',12);

% variance explained
subplot(Nr,Nc,Nc+(1:Nc)); hold on;
errorbar(nanmean(Vexp),nanstd(Vexp)/sqrt(nfolds),'ok','linewidth',3);
plot(bestmodel,mean(Vexp(:,bestmodel)),'.r','markersize',20);
plot(0.5:nModels+0.5,zeros(nModels+1,1),'--k','linewidth',2);
set(gca,'fontsize',16); box off;
set(gca,'XLim',[0 nModels+1]); set(gca,'XTick',1:nModels);
set(gca,'XTickLabel',modellabel);
ylabel('Fraction of variance explained','Fontsize',12);

% tuning functions
for i=1:nvars
    if strcmp(vartype{i},'2D') && ~isempty(fvals{i})
        subplot(Nr,Nc,2*Nc+i);
        imagesc(xvals{i}{1},xvals{i}{2},fvals{i});
        xlabel(varname{i}{1}); ylabel(varname{i}{2});
        set(gca,'fontsize',16); box off;
    elseif ~isempty(fvals{i})
        subplot(Nr,Nc,2*Nc+i);
        plot(xvals{i},fvals{i},'Linewidth',2,'Color','k');
        xlabel(varname{i}); ylabel('Firing rate (spk/s)');
        set(gca,'fontsize',16); box off;
    end
end