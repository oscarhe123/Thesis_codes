clear all 
close all
clc

param = getparam('L11-5V');

xs = [1.7 1.3 0.7 0 -0.7 -1.3 -1.7 0 -1 -0.97]*1e-2; % in m
zs = [2.8 3.2 3.5 3.6 3.5 3.2 2.8 2 0.8 0.77]*1e-2; % in m

RC = ones(size(xs));

tilt = linspace(-4,4,9)/180*pi; % tilt angles in rad
txdel = cell(9,1); % this cell will contain the transmit delays

for k = 1:9
    txdel{k} = txdelay(param,tilt(k));
end

[xi,zi] = meshgrid(linspace(-4e-2,4e-2,100),linspace(0,8e-2,100));

yi = zeros(size(xi));
%%
P = pfield(xi,yi,zi,txdel{5},param);



imagesc(xi(1,:)*1e2,zi(:,1)*1e2,20*log10(P/max(P(:))))
caxis([-30 0]) % dynamic range = [-30,0] dB
c = colorbar;
c.YTickLabel{end} = '0 dB';
colormap([1-hot; hot])
set(gca,'XColor','box','off')
axis equal ij
ylabel('[cm]')
title('The 5^{th} plane wave - RMS pressure field')

%% simulating points 

RF = cell(9,1); % this cell will contain the RF series
param.fs = 4*param.fc; % sampling frequency in Hz

option.WaitBar = false; % remove the progress bar of SIMUS
parfor k = 1:9
    RF{k} = simus(xs,zs,RC,txdel{k},param,option);
end


rf = RF{1}(:,64);
t = (0:numel(rf)-1)/param.fs*1e6; % time (ms)
plot(t,rf)
set(gca,'YColor','none','box','off')
xlabel('time (\mus)')
title('RF signal of the 64^{th} element (1^{st} series, tilt = -4{\circ})')
axis tight

%% demodulation

IQ = cell(9,1);  % this cell will contain the I/Q series

for k = 1:9
    IQ{k} = rf2iq(RF{k},param.fs,param.fc);
end

iq = IQ{1}(:,64);
plot(t,real(iq),t,imag(iq))
set(gca,'YColor','none','box','off')
xlabel('time (\mus)')
title('I/Q signal of the 64^{th} element (1^{st} series, tilt = -10{\circ})')
legend({'in-phase','quadrature'})
axis tight

%% DAS beamforming
param.fnumber = [];

[xi,zi] = meshgrid(linspace(-2e-2,2e-2,1000),linspace(0,4e-2,600));

bIQ = zeros(600,1000,9);  % this array will contain the 21 I/Q images

h = waitbar(0,'');
for k = 1:9
    waitbar(k/9,h,['DAS: I/Q series #' int2str(k) ' of 9'])
    bIQ(:,:,k) = das(IQ{k},xi,zi,txdel{k},param);
    
end
close(h)

I = bmode(bIQ(:,:,1),40); % log-compressed image
imagesc(xi(1,:)*1e2,zi(:,1)*1e2,I)
colormap gray
title('PW-based echo image with a tilt angle of -9{\circ}')

axis equal ij
set(gca,'XColor','none','box','off')
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-40 dB','0 dB'};
ylabel('[cm]')

%% compounding 

cIQ = sum(bIQ,3); % this is the compound beamformed I/Q
I = bmode(cIQ,40); % log-compressed image
imagesc(xi(1,:)*1e2,zi(:,1)*1e2,I)
colormap gray
title('Compound PW-based echo image')

axis equal ij
set(gca,'XColor','none','box','off')
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-40 dB','0 dB'};
ylabel('[cm]')








