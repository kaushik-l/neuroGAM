function [Model,X,Xtype,Nprs,Lambda] = DefineModels(nvars_total,nvars_choose,varchoose,x,xtype,nprs,lambda)

ModelCombo = arrayfun(@(k) mat2cell(nchoosek(1:nvars_total,k),ones(nchoosek(nvars_total,k),1)),nvars_choose,'UniformOutput',false);
ModelCombo = vertcat(ModelCombo{:});
ValidCombos = cellfun(@(x) isempty(setdiff(find(varchoose),x)),ModelCombo);
ModelCombo = ModelCombo(ValidCombos); nModels = length(ModelCombo);
X = cell(nModels,1);
Xtype = cell(nModels,1);
Nprs = cell(nModels,1);
Lambda = cell(nModels,1);
Model = cell(nModels,1); for i=1:nModels, Model{i} = false(1,nvars_total); end
for i=1:nModels
    Model{i}(ModelCombo{i})=true;
    X{i} = cell2mat(x(Model{i})); % X{i} stores inputs for the i^th model
    Xtype{i} = xtype(Model{i});
    Nprs{i} = cell2mat(nprs(Model{i}));
    Lambda{i} = lambda(Model{i});
end