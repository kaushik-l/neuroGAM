function models = FitModels(Model,X,Xtype,Nprs,yt,dt,h,nfolds,Lambda,linkfunc,invlinkfunc)

fprintf(['...... Fitting ' linkfunc '-link model\n']);
nModels = length(Model);
models.class = Model; models.testFit = cell(nModels,1); models.trainFit = cell(nModels,1); models.wts = cell(nModels,1);
for n = 1:nModels
    fprintf('\t- Fitting model %d of %d\n', bin2dec(num2str(Model{n})), bin2dec(num2str(ones(1,length(Model{n})))));
    [models.testFit{n},models.trainFit{n},models.wts{n}] = FitModel(X{n},Xtype{n},Nprs{n},yt,dt,h,nfolds,Lambda{n},linkfunc,invlinkfunc);
end