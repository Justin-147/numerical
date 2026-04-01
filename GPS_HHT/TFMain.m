function [FFT_t,FFT_f,FFT_z,WT_t,WT_f,WT_z,HHT_t,HHT_f,HHT_z,IMF_Matrix,IMF_Amp_Matrix,IMF_Fre_Matrix] = TFMain(Data_ori)

% General Parameter
TF_mode=3;         % 1(STFT), 2(WT), 3(HHT), 4(all of above), 5(FFT)
display_mode=6;    % 1(STFT), 2(WT), 3(HHT), 4(all of above), 5(FFT)
Sampling_rate=1/86400;   % How many observation times within in a second
Freq_max=1/86400/(365*0.5);      % The maximum frequency for the plot  =  Sampling_rate/2
Freq_min=1/86400/365/100;    % The minimum frequency for the plot  =  Sampling_rate*(length(data)/2)
tune_X=1/86400/365;          % The scale of the time
tune_Y=86400;          % The scale of the frequency

% Short Time Fourier Transform Parameter
window=1;             % 1(Rectangular), 2(Hamming), 3(Hanning), 4(Blackman_Tukey)
window_length=3000;     % window_length = 每一次資料處理的長度
step_dist=5;          % step_dist 每次計算移動距離
Tune_win=(1/Sampling_rate*length(Data_ori))/(2)/(1/Freq_max);

% Wavelet Transform Parameter
Basewave=1;           % 1(Morlet), 2(Paul), 3(Dog)

% Hilbert-Huang Transform Parameter
IMF_number=50;        % IMF number
shift_time=2000;       % shifting times of each IMF 
noise=0;          % noise scale = 0.001 (default)
resolution=1/86400/365/100;   % The resolution for the plot & >= Freq_min


[FFT_t,FFT_f,FFT_z,WT_t,WT_f,WT_z,HHT_t,HHT_f,HHT_z,IMF_Matrix,IMF_Amp_Matrix,IMF_Fre_Matrix] = TFATA(Data_ori,TF_mode,display_mode,Sampling_rate,tune_X,tune_Y,window,window_length,step_dist,Tune_win,Basewave,Freq_min,Freq_max,resolution,IMF_number,shift_time,noise);