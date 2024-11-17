% active RIS
function [Rate_DL,Rate_OPT,P]=Active_RIS_2(L,My,Mz,Myy,Mzz,M_bar,M_bar1,K_DL,Pt,kbeams,Training_Size)
%% System Model Parameters

params.scenario='O1_28'; % DeepMIMO Dataset scenario: http://deepmimo.net/
params.active_BS=[1, 2, 3, 4]; % active basestation(/s) in the chosen scenario
D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6; % Bandwidth
Ur_rows = [575 775]; % user Ur rows
r=100;
Validation_Size = 6200; % Validation dataset Size
K = 512; % number of subcarriers
miniBatchSize  = 500; % Size of the minibatch for the Deep Learning
% Note: The axes of the antennas match the axes of the ray-tracing scenario
Mx = 1;  % number of LIS reflecting elements across the x axis
M = Mx.*My.*Mz; % Total number of LIS reflecting elements 
M1 = Mx.*Myy.*Mzz; % Total number of LIS reflecting elements 
Rate_DL = zeros(1,length(Training_Size)); 
Rate_OPT = Rate_DL;
LastValidationRMSE = Rate_DL;
Gt=3;             % dBi
Gr=3;             % dBi
NF=5;             % Noise figure at the User equipment
Process_Gain=10;  % Channel estimation processing gain
noise_power_dB=-204+10*log10(BW/K)+NF-Process_Gain; % Noise power in dB
SNR=10^(.1*(-noise_power_dB))*(10^(.1*(Gt+Gr+Pt)))^2; % Signal-to-noise ratio
% channel estimation noise
noise_power_bar=10^(.1*(noise_power_dB))/(10^(.1*(Gt+Gr+Pt))); 
sigma2=10^(.1*-174);
No_user_pairs = (Ur_rows(2)-Ur_rows(1))*181; % Number of (Ut,Ur) user pairs            
RandP_all = randperm(No_user_pairs).'; % Random permutation of the available dataset
over_sampling_x=1;            % The beamsteering oversampling factor in the x direction
over_sampling_y=1;            % The beamsteering oversampling factor in the y direction
over_sampling_z=1;            % The beamsteering oversampling factor in the z direction
My1=4;    %for bs2
Mz1=1;   % mz1(bs)*mz1(user)=4*8
N=My1*Mz1;
My2=4;    %for bs4
Mz2=1;   % mz1(bs)*mz1(user)=4*8
Pbb=200*10^-3;
Prfchain=40*10^-3;
Pps=10*10^-3;
Plna=20*10^-3;
b=4;

Fs=200*10^6;
FOMw=46.1*10^(-15);
% W=100*10^6;
Padc=FOMw*Fs*2^b;
Pc=M*Pps+M_bar*(Plna+Prfchain+2*Padc)+Pbb;
PBS = db2pow(5); PRIS = db2pow(9);
% P=zeros(1,M/2);
WBS = db2pow(6); WUE = db2pow(-20); WRIS = db2pow(-20); WRA = db2pow(-20);
Knou=1; 
% Generating the BF codebook 
[BF_codebook1]=sqrt(Mx*My*Mz)*UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);   %serving ris
[BF_codebook2]=sqrt(Mx*My1*Mz1)*UPA_codebook_generator(Mx,My1,Mz1,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda); %bs2
  [BF_codebook3]=kron(BF_codebook1,BF_codebook2); %serving
[BF_codebook11]=sqrt(Mx*Myy*Mzz)*UPA_codebook_generator(Mx,Myy,Mzz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);    %interfering ris
%  [BF_codebook22]=sqrt(Mx*My2*Mz2)*UPA_codebook_generator(Mx,My2,Mz2,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda); %bs4
%  [BF_codebook33]=kron(BF_codebook11,BF_codebook22); %interfering
% codebook_size11=size(BF_codebook3,2);   %serving
% codebook_size1=size(BF_codebook33,2);    %interfering

%% DeepMIMO Dataset Generation
disp('-------------------------------------------------------------');
disp([' Calculating for K_DL = ' num2str(K_DL)]);          
params.num_ant_x=1;             % Number of the UPA antenna array on the x-axis 
params.num_ant_y= My ;            % Number of the UPA antenna array on the y-axis 
params.num_ant_z= Mz;             % Number of the UPA antenna array on the z-axis
elements_ris_interf=params.num_ant_y*params.num_ant_z;
params.num_ant_yy= Myy;            % Number of the UPA antenna array on the y-axis 
params.num_ant_zz= Mzz;             % Number of the UPA antenna array on the z-axis
elements_ris_serve=params.num_ant_yy*params.num_ant_zz;
params.num_ant_y1= My1;             % Number of the UPA antenna array on the y-axis 
params.num_ant_z1= Mz1;             % Number of the UPA antenna array on the z-axis
elements_no1=params.num_ant_y1*params.num_ant_z1;    %serving bs2
params.num_ant_y2= My2;             % Number of the UPA antenna array on the y-axis  
params.num_ant_z2= Mz2;             % Number of the UPA antenna array on the z-axis
elements_no2=params.num_ant_y2*params.num_ant_z2;   %interference bs4
params.num_ant_y3= 1;             % Number of the UPA antenna array on the y-axis of the user
params.num_ant_z3= 1;             % Number of the UPA antenna array on the z-axis of the user
elements_no_user=params.num_ant_y3*params.num_ant_z3;
%when bs elements < ris element
  pad=elements_ris_serve*elements_no1-elements_ris_serve*elements_no_user;
%when bs elements > ris element
%  pad=elements_ris_serve*elements_no1-elements_no1*elements_no_user;
  pad1=elements_ris_interf*elements_no2-elements_no2*elements_no_user;
disp('======================================================================================================================');
disp([' Calculating for M = ' num2str(M)]);
Rand_M_bar_all = randperm(M);
Rand_M_bar_all1 = randperm(M1);
   %%%%%%%%%%%%%%%%new active ris1 constructive)
%     BF_codethetabook1_RIS = BF_codebook1;
  BF_codebook33=repmat(BF_codebook11,N,N);
%     BF_codebook33=padarray(BF_codebook11, [pad pad],0,'post');
% BF_codebook3=padarray(BF_codebook1, [pad pad],0,'past');
codebook_size1=size(BF_codebook33,2);    %interfering

    BF_codebook11_RIS2 =BF_codebook33;
    W = exp(1j*2*pi*rand(N*M,K_DL))*sqrt(PBS/1/N);
    wk_temp = reshape(W(:,1),M*N,1);
    w=exp(1j*2*pi*rand(N*M,36))*sqrt(PBS/1/N);
%    W= padarray(W1,[pad 0],0,'post');
            %%% active RIS is fully connected if T=1, then each T RIS elements will
      %%% be served by one power amplifier, otherwise, the active RIS wil
      %%% be subconnected
%   T1 = 1;
T1 = 2; L1=N/T1; zeta=1.1;
C1 = kron(eye(M*N/T1), ones(T1, 1));
     
      C1_mat = inv(C1'*C1)*C1';
%       C1_mat=repmat(C_mat,M,M);
%    Theta = diag(exp(1j*2*pi*rand(M,1)));
   theta1 =BF_codebook3; %but without diag;(must edit in codebook file)
   Theta = 1000*theta1;    
   theta_phase = angle(Theta);
    theta_amp = abs(Theta); 
    theta_amp = C1_mat*theta_amp;
    theta = C1*theta_amp.*exp(1j*theta_phase);
%     theta=diag(Theta);
    BF_codethetabook1_active = theta;

%     BF_codethetabook1_active = padarray(theta, [pad pad], 'post');
  params.ant_spacing=D_Lambda;          % ratio of the wavelnegth; for half wavelength enter .5        
params.bandwidth= BW*1e-9;            % The bandiwdth in GHz 
params.num_OFDM= K;                   % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;        % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=K_DL*1;         % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels
params.num_paths=L;               % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path
params.saveDataset=0;
disp([' Calculating for L = ' num2str(params.num_paths)]);
DeepMIMO_dataset=DeepMIMO_generator(params);
Ht = single(DeepMIMO_dataset{4}.basestation{3}.channel);
B = reshape(Ht,elements_ris_interf*elements_no2,[]);
clear DeepMIMO_dataset
DeepMIMO_dataset=DeepMIMO_generator(params);
Ht1 = single(DeepMIMO_dataset{2}.basestation{1}.channel);
B1 = reshape(Ht1,elements_ris_serve*elements_no1,[]);
clear DeepMIMO_dataset
Validation_Ind = RandP_all(end-Validation_Size+1:end);
[~,VI_sortind] = sort(Validation_Ind);
[~,VI_rev_sortind] = sort(VI_sortind);
%initialization
Ur_rows_step = r; % access the dataset 100 rows at a time
Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2);
Delta_H_max = single(0);
%Delta_H_max1 = single(0);
for pp = 1:1:numel(Ur_rows_grid)-1 % loop for Normalizing H
     clear DeepMIMO_dataset
    params.active_user_first=Ur_rows_grid(pp);
    params.active_user_last=Ur_rows_grid(pp+1)-1;
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params);

    for u=1:params.num_user
        Hr =single(conj(DeepMIMO_dataset{3}.user{u}.channel));    
      Hr1 =single(conj(DeepMIMO_dataset{1}.user{u}.channel));
        %   hrsize=size(Hr);
        A2 = reshape(Hr,elements_ris_interf*elements_no_user,[]);
        A=repmat(A2,elements_no2,1);
         A22 = reshape(Hr1,elements_ris_serve*elements_no_user,[]);
        A1=repmat(A22,elements_no1,1);
%            A= A(:, 1:u_step);
      %  asize=size(A);
     end 
   
     clear DeepMIMO_dataset
    params.active_user_first=Ur_rows_grid(pp);
    params.active_user_last=Ur_rows_grid(pp+1)-1;
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params);

    for u=1:params.num_user
        Hr2u = single(conj(DeepMIMO_dataset{2}.user{u}.channel));
        %    Hr2usize= size(Hr2u);
           wx11=reshape(Hr2u,elements_no1*elements_no_user,[]);
%          wxsize= size(wx);
%           wx=repmat(wx1, elements_no,1);
    end
    clear DeepMIMO_dataset
    params.active_user_first=Ur_rows_grid(pp);
    params.active_user_last=Ur_rows_grid(pp+1)-1;
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params);

    for u=1:params.num_user
            Hr4u =single(conj(DeepMIMO_dataset{4}.user{u}.channel));
       %    Hr4usize= size(Hr4u);
           vx1=reshape(Hr4u,elements_no2*elements_no_user,[]);
        %  vxsize= size(vx);
%           vx=repmat(vx1, elements_no,1);
    end     
   
%     A2u1=wx11.'*BF_codebook2;
%                A4u1=vx1.'*BF_codebook2;

           A4u1=vx1.';
             A4u = padarray(A4u1,[0 pad1],0,'post');
%             A2u = padarray(A2u1,[0 pad],0,'post');
          H_BF=B.*A;
            H_BF1=B1.*A1;
          SNR_sqrt_var=(H_BF.'*BF_codebook11_RIS2);
            SNR_sqrt_var11=H_BF1.'*BF_codethetabook1_active;
            SNR_sqrt_var1=SNR_sqrt_var11.*W';
%             active1=abs(A22.'*BF_codethetabook1_RIS').^2.*sigma2;
  a22=padarray(A22,[pad 0],0,'post');          
active=(a22.'*BF_codethetabook1_active').^2.*sigma2;
          
           %%%%%%%%%%%%%%%%%%parameters of subconnected and fully connected
        wx1= padarray(wx11,[pad1 0],0,'post');
           wx=wx1'.*W'; 
%          temp3= (wx1+ SNR_sqrt_var1').'.*W'; 
          temp3= wx+ SNR_sqrt_var1; 
                      SINR = (SNR*(abs(temp3).^2))./((SNR*(abs(A4u+SNR_sqrt_var)).^2+norm(active))+1);

%             SINR = (SNR*(abs(temp3).^2))./((SNR*((abs(A4u+SNR_sqrt_var)).^2+norm(active)))+1);
%             SNR_sqrt_var1
%  SINR = (SNR*((abs(A2u)+(SNR_sqrt_var1)).^2))./(SNR*((abs(A4u)-(SNR_sqrt_var)).^2)+1);
%       SINR = (SNR*((abs(A2u)+(SNR_sqrt_var1))).^2)./((SNR*((abs(A4u)+(SNR_sqrt_var)).^2+active))+1);

        Delta_H = max(max(abs(SINR)));
        if Delta_H >= Delta_H_max
            Delta_H_max = single(Delta_H);
        end   
     
end
clear Delta_H
disp('=============================================================');
disp([' Calculating for M_bar = ' num2str(M_bar)]);          
Rand_M_bar =unique(Rand_M_bar_all(1:M_bar));
Rand_M_bar1 =unique(Rand_M_bar_all1(1:M_bar1));

%bsize=size(B)
Ht_bar = reshape(B(Rand_M_bar,:),M_bar*K_DL,1);
Ht_bar1 = reshape(B1(Rand_M_bar1,:),M_bar1*K_DL,1);
% htbarsize=size(Ht_bar);
 DL_input1 = single(zeros(M_bar*K_DL*2,No_user_pairs));
 DL_input2 = single(zeros(M_bar1*K_DL*2,No_user_pairs));

DL_output = single(zeros(No_user_pairs,codebook_size1));
DL_output_un=  single(zeros(numel(Validation_Ind),codebook_size1));
% DL_output1 = single(zeros(No_user_pairs,codebook_size1));
% DL_output_un1=  single(zeros(numel(Validation_Ind),codebook_size1));
Delta_H_bar_max = single(0);
Delta_H_bar_max1 = single(0);
count=0;
for pp = 1:1:numel(Ur_rows_grid)-1
    clear DeepMIMO_dataset 
    disp(['Starting received user access ' num2str(pp)]);
    params.active_user_first=Ur_rows_grid(pp);
    params.active_user_last=Ur_rows_grid(pp+1)-1;
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
    %% Construct Deep Learning inputs
    u_step=r;
    Htx=repmat(B(:,1),1,u_step);
        Htx1=repmat(B1(:,1),1,u_step);
b1 = reshape(B1(:,1),M*N,1);
%    Htx=B;
    Hrx=zeros(elements_ris_interf*elements_no_user*elements_no2,u_step);
    Hrx1=zeros(elements_ris_serve*elements_no_user*elements_no1,u_step);
    wx1=zeros(elements_no1*elements_no_user,u_step);
    vxx=zeros(u_step,elements_no2*elements_no_user);
%     
    for u=1:u_step:params.num_user                        
        for uu=1:1:u_step
%              clear DeepMIMO_dataset
%     params.active_user_first=Ur_rows_grid(pp);
%     params.active_user_last=Ur_rows_grid(pp+1)-1;
%     [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
            Hr = single(conj(DeepMIMO_dataset{3}.user{u+uu-1}.channel));
                      Hr1 = single(conj(DeepMIMO_dataset{1}.user{u+uu-1}.channel));

            % hrsize=size(Hr)
           
            A2 = reshape(Hr,elements_ris_interf*elements_no_user,[]);
             %A=repmat(A1(:,:),elements_no1,1);
           %  A1= repmat(A1(:,1:K_DL),1,1);
 A=repmat(A2,elements_no2,1);
             A22 = reshape(Hr1,elements_ris_serve*elements_no_user,[]);
 A1=repmat(A22,elements_no1,1);
          % sizeA=size(A);
%             clear DeepMIMO_dataset
%     params.active_user_first=Ur_rows_grid(pp);
%     params.active_user_last=Ur_rows_grid(pp+1)-1;
%     [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
             Hr2u = single(conj(DeepMIMO_dataset{2}.user{u+uu-1}.channel));
          
             %  Hr2usize= size(Hr2u);
           wx11=reshape(Hr2u,elements_no1*elements_no_user,[]);
%             wxx1=wx1.'*BF_codebook2;
            
            wx1= padarray(wx11,[pad1 0],0,'post');
%          wxsize= size(wx);
        %  wx=repmat(wx1, elements_no,1);
%          clear DeepMIMO_dataset
%     params.active_user_first=Ur_rows_grid(pp);
%     params.active_user_last=Ur_rows_grid(pp+1)-1;
%     [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
            Hr4u =single(conj(DeepMIMO_dataset{4}.user{u+uu-1}.channel));
          % Hr4usize= size(Hr4u);
           vx1=reshape(Hr4u,elements_no2*elements_no_user,[]);
%             vxx1=vx1.'*BF_codebook22;
             vxx1=vx1.';
        %  vxsize= size(vx);
          %vx=repmat(vx1, elements_no,1);


            Hr_bar = reshape(A(Rand_M_bar,:),M_bar*K_DL,1);
                       Hr_bar1 = reshape(A1(Rand_M_bar1,:),M_bar1*K_DL,1);

            %  hrbarsize=size(Hr_bar);

            %--- Constructing the sampled channel
            n1=sqrt(noise_power_bar/2)*(randn(M_bar1*K_DL,1)+1j*randn(M_bar1*K_DL,1));
        %   nisize=size(n1);
            n2=sqrt(noise_power_bar/2)*(randn(M_bar1*K_DL,1)+1j*randn(M_bar1*K_DL,1));
           H_bar = ((Ht_bar+n1).*(Hr_bar+n2));
             H_bar1 = ((Ht_bar1+n1).*(Hr_bar1+n2));
             DL_input1(:,u+uu-1+((pp-1)*params.num_user))= reshape([real(H_bar) imag(H_bar)].',[],1);
            Delta_H_bar = max(max(abs(H_bar)));
             DL_input2(:,u+uu-1+((pp-1)*params.num_user))= reshape([real(H_bar1) imag(H_bar1)].',[],1);
            Delta_H_bar1 = max(max(abs(H_bar1)));
            if Delta_H_bar >= Delta_H_bar_max
                Delta_H_bar_max = single(Delta_H_bar);
            end
              if Delta_H_bar1 >= Delta_H_bar_max1
                Delta_H_bar_max1 = single(Delta_H_bar1);
             end
            Hrx(:,uu)=A(:,1);
             Hrx1(:,uu)=A1(:,1);
%              wxx(uu,:)=wxx1(1,:);
             vxx(uu,:)=vxx1(1,:);
        end
        %--- Actual achievable rate for performance evaluation
        H = Htx.*Hrx;
         H1 = Htx1.*Hrx1;
       % H=reshape(H2, M, []);
        H_BF=H.'*BF_codebook11_RIS2;
       H_BF1=H1.'*BF_codethetabook1_active;

        SNR_sqrt_var = (H_BF);
         SNR_sqrt_var1 = abs(H_BF1);
        %SNR_sqrt_varsize=size(SNR_sqrt_var)
        
        %important
  A4u = padarray(vxx,[0 pad1],0,'post');
%             A2u = padarray(wxx,[0 pad],0,'post');
             wxx = padarray(wx1,[0 36],0,'post');
             W1 = [W w];
             a22=padarray(A22,[pad 36],0,'post');
            active=(a22.'*BF_codethetabook1_active').^2.*sigma2;
            
% A2u= wxx.';
 %A4u=vxx.';
   H_BF11=H_BF1.*W1';
    wXx=wxx'.*W1';
 temp3= wXx+H_BF11;
             SINR = (SNR*(abs(temp3).^2))./((SNR*(abs(A4u+SNR_sqrt_var)).^2+norm(active))+1);

%             SINR = (SNR*(abs(temp3).^2))./((SNR*((abs(A4u+SNR_sqrt_var)).^2+norm(active)))+1);
thetaEE = diag(theta);


%   for T=2:2:8 
%       L1=N/T;
%   i=T/2;
 P = 2*zeta*norm(W/K_DL, 'fro')^2+zeta*(norm(thetaEE.*b1.*wk_temp)^2+norm(thetaEE)^2*sigma2)+Knou*WUE+WBS+M*WRIS+L1*WRA+Pc;
% P = 2*zeta*norm(W/K_DL, 'fro')^2+zeta*(norm(thetaEE.*b1.*wk_temp)^2+norm(thetaEE)^2*sigma2)+M*(WRIS+Pdc)+Pc;

%   end

%   SINR = (SNR*((abs(A2u)+(SNR_sqrt_var1)).^2))./(SNR*((abs(A4u)+(SNR_sqrt_var)).^2+active)+1);
   %   SINR = (SNR*(((A4u)+(SNR_sqrt_var))).^2)./((SNR*(((A2u)+(SNR_sqrt_var1)).^2))+1);

       for uu=1:1:u_step
            if sum((Validation_Ind == u+uu-1+((pp-1)*params.num_user)))
                count=count+1;
            %     DL_output_un(count,:) = single(sum(log2(1+(SNR*(((abs(A4u(uu,:))).^2)./((((abs(A2u(uu,:)).^2+((SNR_sqrt_var(uu,:)).^2)./SNR)))+1)).^2))),1);
%  SINR = ((abs(A4u)).^2)./((((abs(A2u).^2+((SNR_sqrt_var).^2)./SNR)))+1);
% DL_output_un(count,:) = single(sum(log2(1+(SNR*((SNR_sqrt_var(uu,:)).^2))),1));
         DL_output_un(count,:) = single(sum(log2(1+((abs(SINR(uu,:))))),1));
            end
        end
        %--- Label for the sampled channel
                R = single(log2(1+(abs(SINR)/Delta_H_max)));

     %   R = single(log2(1+((SINR)/Delta_H_max).^2));
        % --- DL output normalization
        Delta_Out_max = max(R,[],2);
        if ~sum(Delta_Out_max == 0)
           Rn=diag(1./Delta_Out_max)*R; 
        end
        DL_output(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1,:) = 1*Rn; %%%%% Normalized %%%%%
    end
end
clear u Delta_H_bar R Rn
%-- Sorting back the DL_output_un
DL_output_un = DL_output_un(VI_rev_sortind,:);
%--- DL input normalization 

%  net = configure(net,DL_input);
% net = train(net,DL_input,t);
% view(net)
 DL_input11= 1*(DL_input1/Delta_H_bar_max); %%%%% Normalized from -1->1 %%%%%
 DL_input22= 1*(DL_input2/Delta_H_bar_max1); 
%  DL_input = {DL_input11;DL_input22};
% net = feedforwardnet;
% net.numinputs = 2;
 DL_input = horzcat(DL_input11, DL_input22);

%% DL Beamforming

% ------------------ Training and Testing Datasets -----------------%
DL_output_reshaped = reshape(DL_output.',1,1,size(DL_output,2),size(DL_output,1));
DL_output_reshaped_un = reshape(DL_output_un.',1,1,size(DL_output_un,2),size(DL_output_un,1));
% DL_output_reshaped1 = reshape(DL_output1.',1,1,size(DL_output1,2),size(DL_output1,1));
% DL_output_reshaped_un1 = reshape(DL_output_un1.',1,1,size(DL_output_un1,2),size(DL_output_un1,1));
DL_input_reshaped= reshape(DL_input,size(DL_input,1),1,1,size(DL_input,2));
% DL_input_reshaped1= reshape(DL_input22,size(DL_input22,1),1,1,size(DL_input22,2));
for dd=1:1:numel(Training_Size)
  
    disp([' Calculating for Dataset Size = ' num2str(Training_Size(dd))]);
    Training_Ind   = RandP_all(1:Training_Size(dd));

    XTrain = single(DL_input_reshaped(:,1,1,Training_Ind)); 
    YTrain = single(DL_output_reshaped(1,1,:,Training_Ind));
    XValidation = single(DL_input_reshaped(:,1,1,Validation_Ind));
    YValidation = single(DL_output_reshaped(1,1,:,Validation_Ind));
    YValidation_un = single(DL_output_reshaped_un);

    % ------------------ DL Model definition -----------------%
    layers = [
        imageInputLayer([size(XTrain,1),1,1],'Name','input')

        fullyConnectedLayer(size(YTrain,3),'Name','Fully1')
        reluLayer('Name','relu1')
        dropoutLayer(0.5,'Name','dropout1')

        fullyConnectedLayer(4*size(YTrain,3),'Name','Fully2')
        reluLayer('Name','relu2')
        dropoutLayer(0.5,'Name','dropout2')


        fullyConnectedLayer(4*size(YTrain,3),'Name','Fully3')
        reluLayer('Name','relu3')
        dropoutLayer(0.5,'Name','dropout3')


        fullyConnectedLayer(size(YTrain,3),'Name','Fully4')
        regressionLayer('Name','outReg')];
% analyzeNetwork(layers);
    if Training_Size(dd) < miniBatchSize
        validationFrequency = Training_Size(dd);
    else
        validationFrequency = floor(Training_Size(dd)/miniBatchSize);
    end
    VerboseFrequency = validationFrequency;
    options = trainingOptions('sgdm', ...   
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',20, ...
        'InitialLearnRate',1e-1, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateDropPeriod',3, ...
        'L2Regularization',1e-4,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XValidation,YValidation}, ...
        'ValidationFrequency',validationFrequency, ...
        'Plots','none', ... % 'training-progress'
        'Verbose',0, ...    % 1  
        'ExecutionEnvironment', 'cpu', ...
        'VerboseFrequency',VerboseFrequency);
%     whos

    % ------------- DL Model Training and Prediction -----------------%
    [~,Indmax_OPT]= max(YValidation,[],3);
    Indmax_OPT = squeeze(Indmax_OPT); %Upper bound on achievable rates
    MaxR_OPT = single(zeros(numel(Indmax_OPT),1));                      
    [trainedNet,traininfo]  = trainNetwork(XTrain,YTrain,layers,options);               
    YPredicted = predict(trainedNet,XValidation);

    % --------------------- Achievable Rate --------------------------%                    
    [~,Indmax_DL] = maxk(YPredicted,kbeams,2);
    MaxR_DL = single(zeros(size(Indmax_DL,1),1)); %True achievable rates    
    for b=1:size(Indmax_DL,1)
        MaxR_DL(b) = max(squeeze(YValidation_un(1,1,Indmax_DL(b,:),b)));
        MaxR_OPT(b) = squeeze(YValidation_un(1,1,Indmax_OPT(b),b));
    end
    Rate_OPT(dd) = mean(MaxR_OPT);          
    Rate_DL(dd) = mean(MaxR_DL);
    LastValidationRMSE(dd) = traininfo.ValidationRMSE(end);                                          
    clear trainedNet traininfo YPredicted
    clear layers options Rate_DL_Temp MaxR_DL_Temp Highest_Rate
end              
end
