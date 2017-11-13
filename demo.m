%% example script

%% load data and parameters for demo 1
% load demo1_GiocomoMEC.mat;
% prs = make_demo1params;
% xt = {[posx posy], speed, direction, phase};
% yt = spiketrain;

%% load data and parameters for demo 2
load demo2_AngelakiPPC.mat;
prs = make_demo2params;
xt = {linvel, angvel, dist2target};
yt = spikes;

%% fit and plot
models = BuildGAM(xt,yt,prs); % fit
PlotGAM(models,prs); % plot


function prs = make_demo1params
%% define parameters
prs.varname =  {{'Position-x (cm)' , 'Position-y (cm)'} , 'Speed (cm/s)' , 'Head direction (radian)' , 'Theta phase (radian)'};
prs.vartype = {'2D'  '1D'  '1Dcirc'  '1Dcirc'};
prs.nbins = {[20 , 20] , 10 , 18 , 18};
prs.binrange = [];
prs.nfolds = 10;
prs.dt = 0.02;
prs.filtwidth = 3;
prs.modelname = 'LNP';
prs.lambda = {10 , 50 , 50 , 50};
prs.alpha = 0.05;
end

function prs = make_demo2params
%% define parameters
prs.varname =  {'Linear velocity (cm/s)' , 'Angular Velocity (deg/s)' , 'Distance to target (cm)'};
prs.vartype = {'1D'  '1D'  '1D'};
prs.nbins = {10 , 10 , 10};
prs.binrange = {[0;200], [-90;90], [0;400]};
prs.nfolds = 10;
prs.dt = 0.02;
prs.filtwidth = 15;
prs.modelname = 'LNP';
prs.lambda = {50 , 50 , 50};
prs.alpha = 0.05;
end