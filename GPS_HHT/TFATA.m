function [FFT_t,FFT_f,FFT_z,WT_t,WT_f,WT_z,HHT_t,HHT_f,HHT_z,IMF_Matrix,IMF_Amp_Matrix,IMF_Fre_Matrix] = TFATA(Data_Ori,TF_mode,display_mode,Sampling_rate,tune_X,tune_Y,window,window_length,step_dist,Tune_win,Basewave,Fre_min,Fre_max,resolution,IMF_number,shift_time,noise)
Tune_win;
TF_mode;
FFT_t=[];,FFT_f=[];,FFT_z=[];,
WT_t=[];,WT_f=[];,WT_z=[];,
HHT_t=[];,HHT_f=[];,HHT_z=[];,
l=length(Data_Ori);
 [t,f,z] = TFATA_show(display_mode,Sampling_rate,0,0,0,0,0,0,0,0,0,l,Fre_min, Fre_max,0,0,0,tune_X,tune_Y,Data_Ori,1);


if (TF_mode == 1 | TF_mode == 4) % STFT  Short-Time Fourier Transform
disp('RUN FFT')
[LENX,IMGY,y]= STFT071018 (Data_Ori,Sampling_rate,window,window_length,step_dist,Tune_win);
 y = log10(y);
 [FFT_t,FFT_f,FFT_z] = TFATA_show(display_mode,Sampling_rate,step_dist,LENX,IMGY,y,Tune_win,0,0,0,0,l,Fre_min, Fre_max,0,0,0,tune_X,tune_Y,Data_Ori,2);

%function y = STFT(x, sampling_rate, window, window_length, step_dist, display)
% y = output data
% x = input data
% sampling_rate = observation period ( > 0)
% window = filter by rectangular    = 1
%                    Hamming        = 2
%                    Hanning        = 3
%                    Blackman_Tukey = 4
% window_length = 每一次資料處理的長度
% step_dist 每次計算移動距離
% display figure = 1

end

if (TF_mode == 2 | TF_mode == 4) % MWT  Morlet Wavelet Transform
disp('RUN WT')
[t, frequency, wps, j1, l] = MWT071018(Data_Ori,Sampling_rate,Basewave,Fre_max,Fre_min,Sampling_rate*2);
wps=log10(wps);
[WT_t,WT_f,WT_z] = TFATA_show(display_mode,Sampling_rate,0,0,0,0,0,t, frequency ,wps, j1, l, Fre_min, Fre_max, Sampling_rate*2, 0,0,tune_X,tune_Y,Data_Ori,3);

% z=MWT(x,Fs,Basewave,displayfunc,resolution)
%    Basewave  = 1 : Morlet
%    Basewave  = 2 : Paul
%    Basewave  = 3 : DOG
%    Freqmin = Fmin;%1/(2*(l/Fs))
%    Freqmax = Fmax;%(1/Fs)*1/2
WT_out=wps;
end
if TF_mode == 3 |  TF_mode == 4 % HHT  Hilbert-Huang Transform
disp('RUN HHT')
%[IMF_Matrix,IMF_Fre_Matrix,IMF_Amp_Matrix,IMF_num,Data_len]= HHT071018(AA2,Sampling_rate,1);
%[IMF_Matrix]=emd(AA2,500,2);
[IMF_Matrix]=EEMD071018(Data_Ori,IMF_number,shift_time,noise);
%[E,C]=EEMD(i_n,e_n,h_n_e,nval);
% i_n = input data
% e_n = How many IMFs
% h_n_e = How many times of shifting
% nval = scale of noise, data without noise input zero
IMF_Matrix=IMF_Matrix';
[Data_len,IMF_num]=size(IMF_Matrix);
IMF_num=IMF_num-1;
IMF_sum=sum(IMF_Matrix(:,:),2);
IMF_res=Data_Ori-IMF_sum;

% Hilbert transform
for IMF_loop=1:1:IMF_num
    IMF_Hilb_Matrix(:,IMF_loop)=imag(hilbert(IMF_Matrix(:,IMF_loop)));
    for Data_loop=1:1:Data_len
    IMF_Amp_Matrix(Data_loop,IMF_loop)=sqrt((IMF_Matrix(Data_loop,IMF_loop))^2+(IMF_Hilb_Matrix(Data_loop,IMF_loop))^2);
    IMF_Ang_Matrix(Data_loop,IMF_loop)=atan(IMF_Hilb_Matrix(Data_loop,IMF_loop)/IMF_Matrix(Data_loop,IMF_loop));
    end
    IMF_Fre_Matrix(:,IMF_loop)=gradient(IMF_Ang_Matrix(:,IMF_loop),1/Sampling_rate)/(2*pi);  % due to d*theta/dt is an angle frequency
end

frequency=Fre_min:resolution:Fre_max;
t=1/Sampling_rate:1/Sampling_rate:1/Sampling_rate*l;
 [HHT_t,HHT_f,HHT_z] = TFATA_show(display_mode,Sampling_rate,0,0,0,0,0,t, frequency ,0,0,l, Fre_min, Fre_max, resolution, IMF_Amp_Matrix,IMF_Fre_Matrix,tune_X,tune_Y,Data_Ori,0);
wps=[]; y=[];
HHT_fre_out=IMF_Fre_Matrix;
HHT_amp_out=IMF_Amp_Matrix;
end


if TF_mode == 5 % FFT  Fast Fourier Transform
    [y] = FFT071018 (Data_Ori,Sampling_rate);       
end
