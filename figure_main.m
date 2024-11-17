%% System Model parameters

kbeams=1;   %select the top kbeams, get their feedback and find the max actual achievable rate 
Pt=5; % dB
L =1; % number of channel paths (L)
% Note: The axes of the antennas match the axes of the ray-tracing scenario
My_ar=64; % number of LIS reflecting elements across the y axis
Mz_ar=64; % number of LIS reflecting elements across the z axis
Myy_ar=64; % number of LIS reflecting elements across the y axis
Mzz_ar=64; % number of LIS reflecting elements across the z axis
M_bar1=8; % number of active elements
M_bar=8; % number of active elements
K_DL=64; % number of subcarM1=16, M2=8,N=16riers as input to the Deep Learning model
Training_Size=[2  1e4*(1:.4:3)]; % Training Dataset Size vector

% Preallocation of output variables
 Rate_DLt=zeros(numel(My_ar),numel(Training_Size)); 
 Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));
BW=100*10^6;
%% Figure Data Generation 

for rr = 1:1:numel(My_ar)
    save Fig10_data.mat L My_ar Mz_ar M_bar Training_Size K_DL Rate_DLt Rate_OPTt
 
  [Rate_DL,Rate_OPT]=Active_RIS_2(L,My_ar(rr),Mz_ar(rr),Myy_ar(rr),Mzz_ar(rr),M_bar,M_bar1,K_DL,Pt,kbeams,Training_Size);
  
Rate_DLt(rr,:)=Rate_DL; Rate_OPTt(rr,:)=Rate_OPT;
% EE = Rate_DLt*BW./P;
end


Colour = 'brgmcky';

f = figure('Name', 'Figure', 'units','pixels');
hold on; grid on; box on;
title(['Achievable Rate for different dataset sizes using only ' num2str(M_bar) ' active elements'],'fontsize',12)
xlabel('Deep Learning Training Dataset Size (Thousands of Samples)','fontsize',14)
ylabel('Achievable Rate (bps/Hz)','fontsize',14)
set(gca,'FontSize',13)
if ishandle(f)
    set(0, 'CurrentFigure', f)
    hold on; grid on;
    for rr=1:1:numel(My_ar)
        plot((Training_Size),Rate_OPTt(rr,:),[Colour(rr) '*--'],'markersize',8,'linewidth',2, 'DisplayName',['Genie-Aided Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
        plot((Training_Size),Rate_DLt(rr,:),[Colour(rr) 's-'],'markersize',8,'linewidth',2, 'DisplayName', ['DL Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
%    plot((Training_Size), EE, 'LineWidth', 2, 'Color', 'G');

    end
    legend('Location','SouthEast')
    legend show
end
drawnow
hold off
%save workspace1