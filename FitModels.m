function models = FitModels(Model,x,xtype,nprs,yt,dt,h,nfolds,lambda,linkfunc,invlinkfunc)

nModels = length(Model);
models.class = Model; models.testFit = cell(nModels,1); models.trainFit = cell(nModels,1); models.wts = cell(nModels,1);
for n = 1:nModels
    X = cell2mat(x(Model{n})); % X{i} stores inputs for the i^th model
    Xtype = xtype(Model{n});
    Nprs = cell2mat(nprs(Model{n}));
    Lambda = lambda(Model{n});
    fprintf('\t- Fitting model %d of %d\n', bin2dec(num2str(Model{n})), bin2dec(num2str(ones(1,length(Model{n})))));
    [models.testFit{n},models.trainFit{n},models.wts{n},models.wtsMat{n},models.response{n}] = ...
        FitModel(X,Xtype,Nprs,yt,dt,h,nfolds,Lambda,linkfunc,invlinkfunc);
end