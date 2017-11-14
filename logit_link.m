function [f, df, d2f] = logit_link(param,data,Xtype,Nprs,Lambda)

X = data{1}; % subset of X
Y = data{2}; % number of spikes

% compute the probability of spike
u = X * param;
p = exp(u)./(exp(u) + 1);

% for computing the Hessian
pqX = bsxfun(@times,p.*(1-p),X);

%% Compute roughness penalty
nvars = length(Xtype);
% initialize
J = cell(nvars,1); %f
G = cell(nvars,1); %df
H = cell(nvars,1); %d2f
% compute the contributions to cost(f), gradient(df), and hessian(d2f)
prs = mat2cell(param(:),Nprs,1);
for i=1:length(Nprs)
    [J{i},G{i},H{i}] = roughness_penalty(prs{i},Xtype{i},Lambda{i});
end

%% compute total f, df, and d2f
f = sum(-Y.*log(p) - (1-Y).*log(1-p)) + sum(cell2mat(J));
df = sum(bsxfun(@times,p - Y,X))' + cell2mat(G(:));
d2f = pqX'*X + blkdiag(H{:});

%% functions to compute roughness penalty
function [J,G,H] = roughness_penalty(param,vartype,lambda)
if strcmp(vartype,'2D')
    numParam = numel(param);
    D1 = spdiags(ones(sqrt(numParam),1)*[-1 1],0:1,sqrt(numParam)-1,sqrt(numParam));
    DD1 = D1'*D1;
    M1 = kron(eye(sqrt(numParam)),DD1); M2 = kron(DD1,eye(sqrt(numParam)));
    M = (M1 + M2);
    % compute J, G, and H
    J = lambda*0.5*param'*M*param;
    G = lambda*M*param;
    H = lambda*M;
elseif strcmp(vartype,'1Dcirc')
    numParam = numel(param);
    D1 = spdiags(ones(numParam,1)*[-1 1],0:1,numParam-1,numParam);
    DD1 = D1'*D1;
    % to correct the smoothing across first and last bin
    DD1(1,:) = circshift(DD1(2,:),[0 -1]);
    DD1(end,:) = circshift(DD1(end-1,:),[0 1]);
    % compute J, G, and H
    J = lambda*0.5*param'*DD1*param;
    G = lambda*DD1*param;
    H = lambda*DD1;
elseif strcmp(vartype,'1D')
    numParam = numel(param);
    D1 = spdiags(ones(numParam,1)*[-1 1],0:1,numParam-1,numParam);
    DD1 = D1'*D1;
    % compute J, G, and H
    J = lambda*0.5*param'*DD1*param;
    G = lambda*DD1*param;
    H = lambda*DD1;
end