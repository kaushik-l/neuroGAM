function [x_1hot,xvals,nprs] = Encode1hot(xt,xtype,binrange,nbins)

nt = length(xt);
if strcmp(xtype,'event')
    % initialise with zeros
    binedges = linspace(binrange(1),binrange(2),nbins+1); xvals = 0.5*(binedges(1:end-1) + binedges(2:end));
    x_1hot = zeros(nt,prod(nbins));
    if numel(unique(xt)) == 2
        indx = find(xt);  % identify time indices of the event
        % remove events that are less than kernel_length away from boundary
        indx((indx + binedges(1))<0 | (indx + binedges(end))>nt) = [];
        indx_beg = indx + binedges(1:end-1); % indices of start of ith basis function of kernel
        indx_end = indx + binedges(2:end); % indices of end of ith basis function of kernel
        for i = 1:nbins
            for j=1:length(indx)
                x_1hot(indx_beg(j,i):indx_end(j,i)-1,i) = 1;
            end
        end
    else
        bincenters = 0.5*(binedges(1:end-1) + binedges(2:end));
        xt(1:1 - bincenters(1)) = 0; % remove data around edges so they won't wrap around
        xt(end - bincenters(end):end) = 0;
        for i = 1:nbins, x_1hot(:,i) = circshift(xt,bincenters(i)); end
    end
elseif strcmp(xtype,'2D')
    % initialise with zeros
    binedges1 = linspace(binrange(1,1),binrange(2,1),nbins(1)+1); xvals{1} = 0.5*(binedges1(1:end-1) + binedges1(2:end));
    binedges2 = linspace(binrange(1,2),binrange(2,2),nbins(2)+1); xvals{2} = 0.5*(binedges2(1:end-1) + binedges2(2:end));
    x_1hot = zeros(nt,prod(nbins));
    % identify state index of the ith timebin and set it to 1
    for i = 1:nt
        [~, indx1] = min(abs(xt(i,1)-xvals{1})); [~, indx2] = min(abs(xt(i,2)-xvals{2}));
        indx = sub2ind([nbins(2) nbins(1)], nbins(2) - indx2 + 1, indx1);
        x_1hot(i,indx) = 1;
    end
else
    % initialise with zeros
    binedges = linspace(binrange(1),binrange(2),nbins+1); xvals = 0.5*(binedges(1:end-1) + binedges(2:end));
    x_1hot = zeros(nt,prod(nbins));
    % identify state index of the ith timebin and set it to 1
    for i = 1:nt
        [~, indx] = min(abs(xt(i)-xvals));
        x_1hot(i,indx) = 1;
    end
end
nprs = size(x_1hot,2);