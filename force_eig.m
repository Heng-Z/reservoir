% FORCE.m
%
% This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
% learning rule.
%
% written by David Sussillo
disp('Clearing workspace.');
clear;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
p = 0.3;
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 1440;
dt = 0.1;
learn_every = 2;
Nread = 5;
nRec2Out = N;
noise_level = 0.1;
scale = 1.0/sqrt(p*N);
M = sprandn(N,N,p)*g*scale; % generate random sparse connection with normal distributed weights
M = full(M);%convert the above connection results into matrix


wo = zeros(nRec2Out,Nread); %readout unit sums activities of all neurons
dw = zeros(nRec2Out,Nread);
wf = 2.0*(rand(N,Nread)-0.5); %(-1,1)
wi = rand(N,1)-0.5;
disp(['   N: ', num2str(N)]);
disp(['   g: ', num2str(g)]);
disp(['   p: ', num2str(p)]);
disp(['   nRec2Out: ', num2str(nRec2Out)]);
disp(['   alpha: ', num2str(alpha,3)]);
disp(['   nsecs: ', num2str(nsecs)]);
disp(['   learn_every: ', num2str(learn_every)]);


simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;

%target functions
traces = load('../Stephans/crawl.mat');
ft = traces.tr{1}';
[m,n] = find(isnan(ft));
ft(:,n) = [];
ft = ft./std(ft,0,2);
ft2 = ft(:,length(simtime)+1:2*length(simtime));
ft = ft(:,1:length(simtime));


wo_len = zeros(Nread,simtime_len);    
zt = zeros(Nread,simtime_len);
zpt = zeros(Nread,simtime_len);
x0 = 0.5*randn(N,1); %init state
z0 = 0.5*randn(Nread,1); %init output for feedback

x = x0; 
r = tanh(x);
z = z0; 

figure;
ti = 0;
P0 = (1.0/alpha)*eye(nRec2Out);
P=zeros(nRec2Out,nRec2Out,Nread);
for i = 1:Nread;P(:,:,i) = P0;end

for t = simtime
    ti = ti+1;	
    
    if mod(ti, nsecs/2) == 0
	disp(['time: ' num2str(t,3) '.']);
	subplot 211;
	plot(simtime, ft(1,:), 'linewidth', linewidth, 'color', 'green');
	hold on;
	plot(simtime, zt(1,:), 'linewidth', linewidth, 'color', 'red');
	title('training', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('f', 'z');	
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
	hold off;
	
	subplot 212;
	plot(simtime, wo_len(1,:), 'linewidth', linewidth);
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('|w|', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('|w|');
	pause(0.5);	
    end
    
    % sim, so x(t) and r(t) are created.
    noise = noise_level * randn(N,1);
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt) +wi.*noise*dt; %(x'-x)/dt = -x + M*r +wf*z
    r = tanh(x); %activation?
    z = wo'*r;
    
    if mod(ti, learn_every) == 0
	% update inverse correlation matrix
    for i = 1:Nread
        Pi = P(:,:,i); 
        k = Pi*r;
        rPr = r'*k;
        c = 1.0/(1.0 + rPr);
        Pi = Pi - k*(k'*c);
    
	% update the error for the linear readout
        ei = z(i)-ft(i,ti);
	
	% update the output weights
        dw(:,i) = -ei*k*c;	
        wo(:,i) = wo(:,i) + dw(:,i);
    end
    end
    
    % Store the output of the system.
    zt(:,ti) = z;
    wo_len(:,ti) = sqrt(diag(wo'*wo));	
end

%%
% error_avg = sum(abs(zt-ft))/simtime_len;
% disp(['Training MAE: ' num2str(error_avg,3)]);    
% disp(['Now testing... please wait.']);    


% Now test. 
ti = 0;
for t = simtime				% don't want to subtract time in indices
    ti = ti+1;    
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo'*r;
    
    zpt(:,ti) = z;
end
% error_avg = sum(abs(zpt-ft2))/simtime_len;
% disp(['Testing MAE: ' num2str(error_avg,3)]);
idx = 1;

figure;
subplot 211;
plot(ft(idx,:), 'linewidth', 1, 'color', 'green');
hold on;
plot(zt(idx,:), 'linewidth', 1, 'color', 'red');
title('training', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
hold on;
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z');


subplot 212;
hold on;
plot(ft2(idx,:), 'linewidth', 1, 'color', 'green'); 
axis tight;
plot(zpt(idx,:), 'linewidth', 1, 'color', 'red');
axis tight;
title('simulation', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z');
	

