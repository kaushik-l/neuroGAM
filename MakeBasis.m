function bases = MakeBasis(basistype, nBases, binSize, endPoints)
% Make nonlinearly stretched basis consisting of raised cosines.
% Nonlinear stretching allows faster changes near the event.
%
% 	nBases: [1] - # of basis vectors
%	binSize: time bin size (separation for representing basis
%   endPoints: [2 x 1] = 2-vector containg [1st_peak  last_peak], the peak
%          (i.e. center) of the last raised cosine basis vectors
%   nlOffset: [1] offset for nonlinear stretching of x axis:  y = log(t+nlOffset)
%         (larger nlOffset -> more nearly linear stretching)
%
%  Outputs:  iht = time lattice on which basis is defined
%            ihbasis = basis itself
%            ihctrs  = centers of each basis function
%
%  Example call
%  bases = basisFactory.makeNonlinearRaisedCos(10, 1, [0 500], 2);
if strcmpi(basistype, 'boxcar')
    nbins = diff(endPoints)/binSize;
    iht = (1:nbins)'*binSize + endPoints(1);
    width = floor(nbins/nBases);
    ihbasis = zeros(nbins,nBases); ihbasis(1:width,:) = 1;
    shifts = width*(0:nBases-1);
    ihbasis = bsxfun(@(x,y) circshift(x,y), ihbasis, shifts);
    ihctrs = binSize*((0:nBases-1)*width + width/2) + endPoints(1);
elseif strcmpi(basistype, 'raisedcosine')
    % For raised cosine, the spacing between the centers must be 1/4 of the
    % width of the cosine
    nbins = diff(endPoints)/binSize;
    iht = repmat((1:nbins)', 1, nBases);
    dbcenter = nbins / (3 + nBases); % spacing between bumps
    width = 4 * dbcenter; % width of each bump
    bcenters = 2 * dbcenter + dbcenter*(0:nBases-1); % location of each bump centers
    bfun = @(x,period)((abs(x/period)<0.5).*(cos(x*2*pi/period)*.5+.5));
    ihbasis = bfun(iht-repmat(bcenters,nbins,1), width);
    iht = iht(:,1)*binSize + endPoints(1);
    ihctrs = bcenters*binSize + endPoints(1);
elseif strcmp(basistype,'nlraisedcosine')
    % nonlinearity for stretching x axis (and its inverse)
    nlin = @(x)(log(x + 1e-10));
    invnl = @(x)(exp(x) - 1e-10);
    yrnge = nlin(endPoints + binSize/3);
    db = diff(yrnge) / (nBases-1); % spacing between raised cosine peaks
    ctrs = yrnge(1):db:yrnge(2); % centers for basis vectors
    mxt = invnl(yrnge(2)+2*db) - binSize/3; % maximum time bin
    iht = (0:binSize:mxt)';
    ff = @(x,c,dc) (cos(max(-pi, min(pi, (x-c)*pi/dc/2))) + 1)/2;
    ihbasis = ff(repmat(nlin(iht + binSize/3), 1, nBases), repmat(ctrs, numel(iht), 1), db);
    ihctrs = invnl(ctrs);
end

bases.y = ihbasis;
bases.x = iht;
% bases.c = ihctrs;