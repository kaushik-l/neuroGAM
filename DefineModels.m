function Model = DefineModels(nvars_total,nvars_choose,varchoose)

ModelCombo = arrayfun(@(k) mat2cell(nchoosek(1:nvars_total,k),ones(nchoosek(nvars_total,k),1)),nvars_choose,'UniformOutput',false);
ModelCombo = vertcat(ModelCombo{:});
ValidCombos = cellfun(@(x) isempty(setdiff(find(varchoose),x)),ModelCombo);
ModelCombo = ModelCombo(ValidCombos);
nModels = length(ModelCombo);
Model = cell(nModels,1); for i=1:nModels, Model{i} = false(1,nvars_total); end
for i=1:nModels, Model{i}(ModelCombo{i})=true; end