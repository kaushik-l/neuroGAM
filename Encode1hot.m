function [x_1hot,xvals,nprs] = Encode1hot(xt,xtype,binrange,nbins)

nt = length(xt);
if strcmp(xtype,'2D')
    % initialise with zeros
   binedges1 = linspace(binrange(1,1),binrange(2,1),nbins(1)+1); xvals{1} = 0.5*(binedges1(1:end-1) + binedges1(2:end));
   binedges2 = linspace(binrange(1,2),binrange(2,2),nbins(2)+1); xvals{2} = 0.5*(binedges2(1:end-1) + binedges2(2:end));
   x_1hot = zeros(nt,prod(nbins));
   % identify index of the ith state and set it to 1
   for i = 1:nt
       [~, indx1] = min(abs(xt(i,1)-xvals{1})); [~, indx2] = min(abs(xt(i,2)-xvals{2}));
       indx = sub2ind([nbins(2) nbins(1)], nbins(2) - indx2 + 1, indx1);
       x_1hot(i,indx) = 1;
   end
else
    % initialise with zeros
    binedges = linspace(binrange(1),binrange(2),nbins+1); xvals = 0.5*(binedges(1:end-1) + binedges(2:end));
    x_1hot = zeros(nt,prod(nbins));
    % identify index of the ith state and set it to 1
    for i = 1:nt
        [~, indx] = min(abs(xt(i)-xvals));
        x_1hot(i,indx) = 1;
    end
end
nprs = size(x_1hot,2);