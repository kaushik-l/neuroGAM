function [f, df, d2f] = identity_link(param,data,Xtype,Nprs,Lambda)

X = data{1}; % subset of X
Y = data{2}; % number of spikes

% compute the firing rate
rate = X * param;

% for computing the Hessian
Yinvr2X = bsxfun(@times,Y.*(rate.^-2),X);

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
f = sum(rate-Y.*log(rate)) + sum(cell2mat(J));
df = real(X' * (1 - Y./rate) + cell2mat(G(:)));
d2f = Yinvr2X'*X + blkdiag(H{:});

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
elseif strcmp(vartype,'0D')
    J = lambda*sum(abs(param));
    G = lambda*sign(param);
    H = diag(zeros(1,length(param)));
end