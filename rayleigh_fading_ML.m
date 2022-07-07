clear;
clc;
Q_sym = []; %FOR STORING QAM SYMBOLS
H = []; %FOR STORING QAM MODULATION VALUES OF M

for i=1:1:8
    Q_sym(i) = qammod(i-1,8);
    H(i) = 8;
end
for i=1:1:16
    Q_sym(8+i) = qammod(i-1,16);
    H(8+i) = 16;
end
for i=1:1:32
    Q_sym(16+8+i) = qammod(i-1,32);
    H(16+8+i) = 32;
end


F = Q_sym'; %ENTIRE TRAINING SET CONTAINING MODULATED SYMBOLS IN COMPLEX FORMAT
FR = real(Q_sym)'; %REAL PART OF THE MODULATED SYMBOLS FROM TRAINING DATA
FI = imag(Q_sym)'; %IMAGINARY PART OF THE MODULATED SYMBOLS FROM TRAINING DATA

J = H'; %COLUMN CONTAINING NATURE OF QAM-MODULATION
D_set = table(F); %OUR REQUIRED DATASET FOR NAIVE BAYES
D_set1 = table(FR,FI); % OUR REQUIRED DATASET FOR KNN AND SVM
subscriptObj = vartype('numeric'); % CREATING NUMERIC OBJECT
Numeric_D_set = D_set(:,subscriptObj);% TURNING OUR TABLE INTO NUMERIC MATRIX FOR NAIVE BAYES
Numeric_D_set1 = D_set1(:,subscriptObj);% TURNING OUR TABLE INTO NUMERIC MATRIX FOR KNN AND SVM

result_mdl = fitcnb(Numeric_D_set,J,'Distribution','kernel'); %CREATING NAIVE BAYES CLASSIFICATION MODEL
result_mdl1 = fitcknn(Numeric_D_set1,J,'NumNeighbors',11,'Standardize',1);% CREATING KNN CLASSIFICATION MODEL
t = templateSVM('KernelFunction','polynomial'); %Creating an SVM template, and standardizing the predictors
result_mdl2 = fitcecoc(Numeric_D_set1,J,'learner',t);% CREATING SVM CLASSIFICATION MODEL

%% ............ TRANSMITTER AND RECEIVER.....................
i=imread('image_clg.jpg');
I=rgb2gray(i); % converts to greyscale image
BW=imbinarize(I); % converts to greyscale binary image
BW = imresize (BW,[100 100]);
[BW_r,BW_c]=size(BW);
BW_1D=reshape(BW',1,[]); % 1D binary image = input binary data

M=32; %QAM modulation order
N=length(BW_1D)/log2(M);% no of Mqam modulated symbols
c = 16;
taps = 4;
N_runs = 1;
q=bi2de(reshape(BW_1D,log2(M),N).','left-msb'); % binary to decimal nos

%% TRANSMITTER END
y = qammod(q',M); %32 qam modulator
z = ifft(y,N); %inverse fast fourier transform
z1 = [z(end-c+1:end) z];%addition of cyclic prefix
Eb_N0_dB =0:2:20;
N_err=zeros(1,length(Eb_N0_dB));
N_err1=zeros(1,length(Eb_N0_dB));
N_err2=zeros(1,length(Eb_N0_dB));
N_err3=zeros(1,length(Eb_N0_dB));

Acc_ze1 = zeros(1,length(Eb_N0_dB));
Acc_ze2 = zeros(1,length(Eb_N0_dB));
Acc_ze3 = zeros(1,length(Eb_N0_dB));
%% 
for i_r=1:1:N_runs  
    for in = 1:1:length(Eb_N0_dB)
        %VARIANCE = 0.5 - TRACKS THEORITICAL PDF CLOSELY (FADING CHANNEL
        %COEFFICIENT)
        %% 
        txSig=conj(z1)';
        h = [(1/sqrt(0.5))*(randn(1,taps)+1i*randn(1,taps))  zeros(1,size(txSig,1)-taps)];%Fading coefficient distribution in complex form
        h_t=toeplitz([h(1) fliplr(h(2:end))], h);
        fadedSig=h_t*txSig; %Tx multiplied with Rayleigh fading coefficient (ray)
        rx = awgn(fadedSig,Eb_N0_dB(in),'measured'); %addition of awgn noise with each value of snr 
%------------------------------------------------------------------------------------------------------    
        %% RECEIVER END
        rxSig_eq = inv(h_t'*h_t)*(h_t')*rx; %Equaliser
        noise_ofdm = conj(rxSig_eq)';
        noise_ofdm_ser=reshape(conj(noise_ofdm)',1,[]); 
        noz1r = noise_ofdm(c+1:end);%removal of cyclic prefix
        fz = reshape(noz1r,N,[]).'; %serial to parallel converter
        fz1 = fft(fz,N); %fast fourier transform
        FR1 = real(fz1)'; %REAL PART OF THE MODULATED SYMBOLS FROM TEST DATA
        FI1 = imag(fz1)'; %IMAGINARY PART OF THE MODULATED SYMBOLS FROM TEST DATA

        ofz = qamdemod(fz1,M); %qam demodulation without any prediction
    
        M1 = predict(result_mdl,fz1'); %Predicted modulations using Naive Bayes Model
        M2 = predict(result_mdl1,[FR1,FI1]); %Predicted modulations using KNN model
        M3 = predict(result_mdl2,[FR1,FI1]); %Predicted modulations using SVM model

        for i = 1:1:size(M1)
            ofz1(i) = qamdemod(fz1(i),M1(i)); %qam demodulation with prediction using Naive Bayes     
        end
        for i = 1:1:size(M2)
            ofz2(i) = qamdemod(fz1(i),M2(i)); %qam demodulation with prediction using KNN     
        end
        for i = 1:1:size(M3)
            ofz3(i) = qamdemod(fz1(i),M3(i)); %qam demodulation with prediction uisng SVM    
        end
    
        ofz1r = de2bi(ofz,log2(M),'left-msb');%converting decimal to binary
        ofz1r1 = de2bi(ofz1,log2(M),'left-msb');%converting decimal to binary (Naive Bayes)
        ofz1r2 = de2bi(ofz2,log2(M),'left-msb');%converting decimal to binary (KNN)
        ofz1r3 = de2bi(ofz3,log2(M),'left-msb');%converting decimal to binary (SVM)

        final_received = reshape(ofz1r',1,[]);%RECEIVED BINARY BIT STREAM
        final_received1 = reshape(ofz1r1',1,[]);%RECEIVED BINARY BIT STREAM (Naive Bayes)
        final_received2 = reshape(ofz1r2',1,[]);%RECEIVED BINARY BIT STREAM (KNN)
        final_received3 = reshape(ofz1r3',1,[]);%RECEIVED BINARY BIT STREAM (SVM)
    
        %% Calculating BER
        N_bits = length(BW_1D);
        errors(in) = sum(BW_1D~=final_received);% Number of unequal elements between x & final_received
        errors1(in) = sum(BW_1D~=final_received1);% Number of unequal elements between x & final_received
        errors2(in) = sum(BW_1D~=final_received2);% Number of unequal elements between x & final_received
        errors3(in) = sum(BW_1D~=final_received3);% Number of unequal elements between x & final_received
        
        Acc(in)=(sum(M1'==32)/size(M1,1))*100; %Accuracy values of Naive Bayes for different values of SNR
        Acc1(in)=(sum(M2'==32)/size(M2,1))*100; %Accuracy values of KNN for different values of SNR
        Acc2(in)=(sum(M3'==32)/size(M3,1))*100; %Accuracy values of SVM for different values of SNR
    end %End of SNR Loop
    N_err = N_err+errors;
    N_err1 = N_err1+errors1;
    N_err2 = N_err2+errors2;
    N_err3 = N_err3+errors3;
    
    Acc_ze1 = Acc_ze1 + Acc;
    Acc_ze2 = Acc_ze2 + Acc1;
    Acc_ze3 = Acc_ze3 + Acc2;
end
BER = N_err/N_bits/N_runs
BER1 = N_err1/N_bits/N_runs
BER2 = N_err2/N_bits/N_runs
BER3 = N_err3/N_bits/N_runs

Accuracy1 = Acc_ze1/N_runs
Accuracy2 = Acc_ze2/N_runs
Accuracy3 = Acc_ze3/N_runs

ber_ri = berfading(Eb_N0_dB,'qam',32,5,2);
f1=berfit(Eb_N0_dB,BER);
f2=berfit(Eb_N0_dB,BER1);
f3=berfit(Eb_N0_dB,BER2);
f4=berfit(Eb_N0_dB,BER3);
%-------------------------------------------------------------------

%% PLOTING THE BER VS SNR (dB) CURVE ON LOG SCALE
% BER THROUGH SIMULATION
figure(1);
semilogy(Eb_N0_dB(1:length(f1)),f1,'^r-','Linewidth',2);
hold on
semilogy(Eb_N0_dB,ber_ri,'--m','Linewidth',2);
hold on
xlabel('SNR (dB)');
ylabel('BER');
title('BER v/s SNR plot for Adaptive M-QAM Modulation (where M = 32)');
grid on

figure(1);
semilogy(Eb_N0_dB(1:length(f2)),f2,'--g*','Linewidth',2);
hold on
grid on

figure(1);
semilogy(Eb_N0_dB(1:length(f3)),f3,'--bd','Linewidth',2);
hold on
grid on

figure(1);
semilogy(Eb_N0_dB(1:length(f4)),f4,'--k*','Linewidth',1);
hold on
grid on

%% THEORITICAL BER 
figure(1);
theoryBerAWGN = (0.5).*erfc(sqrt((10.^(Eb_N0_dB/10))));
semilogy(Eb_N0_dB, theoryBerAWGN,'g-+','Linewidth',2);
grid on;

legend('Rayleigh with out Adaptive Modulation','Rician with out Adaptive Modulation','Rayleigh with NB','Rayleigh with KNN','Rayleigh with SVM','AWGN');
axis([Eb_N0_dB(1,1) Eb_N0_dB(end) 0.00001 1]);

%% PLOTING THE ACCURACY VS SNR (dB) 
figure(2);
plot(Eb_N0_dB,Accuracy1,'--bo','Linewidth',2);
hold on
title('ACCURACY VS SNR');
ylabel('ACCURACY(%)');
xlabel('SNR(dB)');
grid on

figure(2);
plot(Eb_N0_dB,Accuracy2,'--r^','Linewidth',2);
hold on
grid on

figure(2);
plot(Eb_N0_dB,Accuracy3,'--ko','Linewidth',2);
grid on
legend('ACCURACY OF NB','ACCURACY OF KNN','ACCURACY OF SVM');

%% Retrieving actual image
Rx_bin_ser=reshape(final_received',1,[]);%% received serial binary bits
Rx_BW=reshape(Rx_bin_ser,BW_c,BW_r).'; % converts back binary image

Rx_bin_ser1=reshape(final_received1',1,[]);%% received serial binary bits
Rx_BW1=reshape(Rx_bin_ser1,BW_c,BW_r).'; % converts back binary image

Rx_bin_ser2=reshape(final_received2',1,[]);%% received serial binary bits
Rx_BW2=reshape(Rx_bin_ser2,BW_c,BW_r).'; % converts back binary image

Rx_bin_ser3=reshape(final_received3',1,[]);%% received serial binary bits
Rx_BW3=reshape(Rx_bin_ser3,BW_c,BW_r).'; % converts back binary image

%% Transmitted image
figure(3);
imshow(BW);
hold on;
title('Transmitted image');
%% Received image
figure(4);
imshow(Rx_BW);
hold on;
title('Received image without ML');

figure(5);
imshow(Rx_BW1);
hold on;
title('Received image with NB');

figure(6);
imshow(Rx_BW2);
hold on;
title('Received image with KNN');

figure(7);
imshow(Rx_BW3);
hold on;
title('Received image with SVM');


Model_Accuracy_NB = mean(Accuracy1)
Model_Accuracy_KNN = mean(Accuracy2)
Model_Accuracy_SVM = mean(Accuracy3)