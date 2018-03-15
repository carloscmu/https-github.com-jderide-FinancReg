%==========================================================================
% Code for "Robust Regulation of Financial Systems"
% Author: CARLOS RAMIREZ
% This version: March 9, 2018.
%==========================================================================

clc; clear all; close all;
cd('/sfm/home/m1car03/SNL-FRB/codes')

%% Parameters of simulation
global TolF Tolx MaxFunE MaxI NSim
TolF = 10^-6; Tolx = 10^-8; MaxFunE = 10^5; MaxI = 10^5; 
NSim = 1*10^4; 

%% Preferences' parameters
global gamma theta
gamma = 2; % investor risk aversion. 
theta = 2; % investors ambiguity aversion

%% Parameters defining distribution of idiosyncratic shocks
global lambda ub a0 a1
ub = .25; lambda = .05; a0 = 1; a1=1;

%% Propagation parameters
global p_array prior p_bar
p_array = [.25; .5; .75]; % propensity of two trees being connected
prior = [.25; .5; .25]; % prior on p 
p_bar = p_array'*prior; 

%% Initial network
adj = [0 1 0; 1 0 1; 0 1 0];
[n ~] =size(adj);

lbx = 0; ubx = .5;
%lbx = lambda; 
%ubx = (lambda/(1-ub))+lambda;
xe = lbx:((ubx-lbx)/50):ubx; %grid for node in the extreme
xc = xe; % grid for node in the middle

tic
for i = 1:1:length(xe)
    for j = 1:1:length(xc)
        x = [xe(i);xc(j);xe(i)];               
        sim = 1;        
        while sim < NSim+1
            index = randi([1 n],1,1); epsilon_ini = zeros(n,1); epsilon_ini(index) = ub.*rand(1); 
            p = p_bar; s = ep(adj,p,ones(n,1)); r = r_optimal(x,s,lambda,ub,a0,a1); %optimal r 
            epsilon = ep(adj,p,epsilon_ini);
            pi(:,sim) = a0 - a1.*(r.*(1-epsilon)); % nodes realized profits
            sp(sim) = sum(pi(:,sim)); % sum of nodes' realized profits
            sim = sim +1;
        end
        ES(i,j) = mean(sp); VS(i,j) = var(sp);
    end
end
toc

% Computing ambiguity term on SP objective function
tic
for i = 1:1:length(xe)
    for j = 1:1:length(xc)
        x = [xe(i);xc(j);xe(i)];  
        
        for l = 1:1:length(p_array)
            p = p_array(l);  
            s = ep(adj,p,ones(n,1)); r = r_optimal(x,s,lambda,ub,a0,a1); %optimal r  
            sim = 1;
            while sim < NSim+1
                index = randi([1 n],1,1); epsilon_ini = zeros(n,1); epsilon_ini(index) = ub.*rand(1); 
                epsilon = ep(adj,p,epsilon_ini);
                pi(:,sim) = a0 - a1.*(r.*(1-epsilon)); % nodes realized profits
                sp(sim) = sum(pi(:,sim)); % sum of nodes' realized profits            
                sim = sim +1;
            end
            ESP(i,j,l) = mean(sp); % E(sp|p)
            %VSP(i,j,l) = var(sp); % V(sp|p)
        end
        
        % index vector that selects p according to the prior distribution
        for k = 1:1:NSim 
            r = rand(1);
            if r < prior(1)
                ind(k,1) = 1;
            elseif r > (1-prior(3))
                ind(k,1) = 3;
            else
                ind(k,1) = 2;
            end
        end
        %ind = randi([1 length(p_array)],NSim,1);        
        VES(i,j) = var(ESP(i,j,ind),0,3);
        %EVSP(i,j) = prior(1)*VSP(i,j,1) + prior(2)*VSP(i,j,2) + prior(3)*VSP(i,j,3); % E(V(sp|p))
    end
end
toc

[X,Y] = meshgrid(xe,xc);
xe_int = min(xe):((max(xe)-min(xe))/100):max(xe); xc_int = min(xc):((max(xc)-min(xc))/100):max(xc);
[X_int,Y_int] = meshgrid(xe_int,xc_int);
ES_int = interp2(X,Y,ES',X_int,Y_int,'spline'); % ES in interpolated values
VS_int = interp2(X,Y,VS',X_int,Y_int,'spline'); % VS in interpolated values
VES_int = interp2(X,Y,VES',X_int,Y_int,'spline'); % VES in interpolated values
%EVSP_int = interp2(X,Y,EVSP',X_int,Y_int,'spline'); % EVSP in interpolated values


figure(1)
subplot(3,2,1)
surf(X_int,Y_int,ES_int)
title('$E_{\bar{p}}[\sum_i \pi^*_i]$','Interpreter','latex')
xlabel('$x_e$','Interpreter','latex'); 
ylabel('$x_c$','Interpreter','latex');
shading flat; colorbar;

subplot(3,2,2)
surf(X_int,Y_int,VS_int)
title('$V_{\bar{p}}[\sum_i \pi^*_i]$','Interpreter','latex')
xlabel('$x_e$','Interpreter','latex'); 
ylabel('$x_c$','Interpreter','latex');
shading flat; colorbar;

subplot(3,2,3)
surf(X_int,Y_int,ES_int-(gamma/2).*VS_int)
title('$E_{\bar{p}}[\sum_i \pi^*_i] - (\gamma/2)*V_{\bar{p}}[\sum_i \pi^*_i]$','Interpreter','latex')
xlabel('$x_e$','Interpreter','latex'); 
ylabel('$x_c$','Interpreter','latex');
shading flat; colorbar;

subplot(3,2,4)
surf(X_int,Y_int,VES_int)%VS_int-EVSP_int) % V(E(Y|X)) = V(Y) - E(V(Y|X))
title('$V_{\mu}(E_{p}[\sum_i \pi^*_i])$','Interpreter','latex')
xlabel('$x_e$','Interpreter','latex'); 
ylabel('$x_c$','Interpreter','latex');
shading flat; colorbar;

subplot(3,2,5:6)
surf(X_int,Y_int,ES_int-(gamma/2).*VS_int - (theta/2).*(VES_int)) % V(E(Y|X)) = V(Y) - E(V(Y|X))
title('$E_{\bar{p}}[\sum_i \pi^*_i] - (\gamma/2)*V_{\bar{p}}[\sum_i \pi^*_i] - (\theta/2)*V_{\mu}(E_{p}[\sum_i \pi^*_i])$','Interpreter','latex')
xlabel('$x_e$','Interpreter','latex'); 
ylabel('$x_c$','Interpreter','latex');
shading flat; colorbar;

ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0  1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
text(.3,.98,['\lambda = ' num2str(lambda) ', \gamma = ' num2str(gamma) ', \theta = ' num2str(theta) ', UB = ' num2str(ub) ', E[p] = ' num2str(p_bar) ', NSim = ' num2str(NSim) ', \mu = [.25 .5 .25] ']);

figure(2)
opy = diag(ES_int-(gamma/2).*VS_int - (2000/2).*(VES_int));
opx = diag(Y_int);
plot(opx,opy)
xlabel('$x= x_c = x_e$','Interpreter','latex');
title('$E_{\bar{p}}[\sum_i \pi^*_i] - (\gamma/2)*V_{\bar{p}}[\sum_i \pi^*_i] - (\theta/2)*V_{\mu}(E_{p}[\sum_i \pi^*_i])$','Interpreter','latex')

