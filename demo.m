%% example script

%% load data and parameters for demo 1
% load demo1_GiocomoMEC.mat;
% prs = make_demo4params;
% xt = {[posx(:) posy(:)], speed(:), direction(:), phase(:)}; % each of the 5 variables must be T x 1 column vector
% yt = spiketrain;

%% load data and parameters for demo 2
load demo2_AngelakiPPC.mat;
prs = make_demo2params;
xt = {linvel(:), angvel(:), dist2target(:)}; % each of the 3 variables must be T x 1 column vector
yt = spikes(:);

%% fit and plot
models = BuildGAM(xt,yt,prs); % fit
PlotGAM(models,prs); % plot

%% funtion definitions
function prs = make_demo1params
%% define parameters
prs.varname =  {{'Position-x (cm)' , 'Position-y (cm)'} , 'Speed (cm/s)' , 'Head direction (radian)' , 'Theta phase (radian)'};
prs.vartype = {'2D'  '1D'  '1Dcirc'  '1Dcirc'};
prs.basistype = {'boxcar'  'boxcar'  'boxcar', 'boxcar'};
prs.nbins = {[20 , 20] , 10 , 18 , 18};
prs.binrange = [];
prs.nfolds = 10;
prs.dt = 0.02;
prs.filtwidth = 3;
prs.linkfunc = 'log';
prs.lambda = {10 , 50 , 50 , 50};
prs.alpha = 0.05;
prs.varchoose = [1,1,1,1];
prs.method = 'fastbackward';
end

function prs = make_demo2params
%% define parameters
prs.varname =  {'Linear velocity (cm/s)' , 'Angular Velocity (deg/s)' , 'Distance to target (cm)'};
prs.vartype = {'1D'  '1D'  '1D'};
prs.basistype = {'boxcar'  'boxcar'  'boxcar'};
prs.nbins = {10 , 10 , 10};
prs.binrange = {[0;200], [-90;90], [0;400]}; % it is crucial to specify the range  if there are outliers in your stimulus
prs.nfolds = 10;
prs.dt = 0.02;
prs.filtwidth = 15;
prs.linkfunc = 'log';
prs.lambda = {50 , 50 , 50};
prs.alpha = 0.05;
prs.varchoose = [0,0,0];
prs.method = 'fastbackward';
end