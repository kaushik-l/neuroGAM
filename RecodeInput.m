function [x_recoded,basis,nprs] = RecodeInput(xt,xtype,binrange,nbins,basistype,dt)

nt = length(xt);
if strcmp(xtype,'event')
    binedges = linspace(binrange(1),binrange(2),nbins+1); xvals = 0.5*(binedges(1:end-1) + binedges(2:end));
    xt(1:1 - binedges(1)) = 0; xt(end - binedges(end):end) = 0; % remove data around edges so they won't wrap around
    x_recoded = zeros(nt,prod(nbins)); % initialise with zeros
    if any(strcmp(basistype,{'boxcar','raisedcosine','nlraisedcosine'}))
        basis = MakeBasis(basistype, nbins, dt, dt*[binedges(1) binedges(end)]);
            for i=1:size(basis.y,2)
                xt_conv = conv(xt,basis.y(:,i));
                x_recoded(:,i) = xt_conv(1:end-numel(basis.x)+1);
            end
            x_recoded = circshift(x_recoded, round(basis.x(1)/dt));
    elseif strcmp(basistype,'delta')
            bincenters = 0.5*(binedges(1:end-1) + binedges(2:end));
            for i = 1:nbins, x_recoded(:,i) = circshift(xt,bincenters(i)); end
    end
elseif strcmp(xtype,'2D')
    % initialise with zeros
    binedges1 = linspace(binrange(1,1),binrange(2,1),nbins(1)+1); xvals{1} = 0.5*(binedges1(1:end-1) + binedges1(2:end));
    binedges2 = linspace(binrange(1,2),binrange(2,2),nbins(2)+1); xvals{2} = 0.5*(binedges2(1:end-1) + binedges2(2:end));
    x_recoded = zeros(nt,prod(nbins));
    % identify state index of the ith timebin and set it to 1
    for i = 1:nt
        [~, indx1] = min(abs(xt(i,1)-xvals{1})); [~, indx2] = min(abs(xt(i,2)-xvals{2}));
        indx = sub2ind([nbins(2) nbins(1)], nbins(2) - indx2 + 1, indx1);
        x_recoded(i,indx) = 1;
    end
    basis.y = eye(nbins);
    basis.x = xvals;
else % if xtype is '1D' or '1Dcirc'
    binedges = linspace(binrange(1),binrange(2),nbins+1); xvals = 0.5*(binedges(1:end-1) + binedges(2:end));
    x_recoded = zeros(nt,prod(nbins));
    % identify state index of the ith timebin and set it to 1
    for i = 1:nt
        [~, indx] = min(abs(xt(i)-xvals));
        x_recoded(i,indx) = 1;
    end
    basis.y = eye(nbins);
    basis.x = xvals(:);
end
nprs = size(x_recoded,2);