function [x_1hot,xvals] = Encode1hot(xt,binedges)

% initialise with zeros
xvals = 0.5*(binedges(1:end-1) + binedges(2:end));
x_1hot = zeros(length(xt),length(xvals));

for i = 1:numel(xt)    
    % identify index of the ith state and set it to 1
    [~, indx] = min(abs(xt(i)-xvals));
    x_1hot(i,indx) = 1;
end