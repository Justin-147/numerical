function [ttt,f,z] =  TFATA_show(display_mode,Sampling_rate,step_dist,LENX,IMGY,y,Tune_win,t, frequency ,wps, j1, l, Fre_min, Fre_max, resolution, Z_Ori, Y_Ori,tune_X,tune_Y,Data_Ori,Show_CK);
ll=1:1:l;
tt=(1/Sampling_rate)*ll*tune_X;
ttt=[];, f=[];, z=[];
if (display_mode >= 1 | display_mode <= 5) & Show_CK == 1
    if display_mode == 4
        subplot(4,4,1:4)
    else
        figure
    end
     plot(tt,Data_Ori)
     title('Raw Data')
     axis([ tt(1) tt(l) -inf inf])
end

if (display_mode == 1 | display_mode == 4)  & step_dist ~=0 & Show_CK == 2
    if display_mode == 1
        figure
    else
        subplot(4,4,5:8)
    end
    Data_len=length(Data_Ori);
    ST=((Data_len/step_dist)-LENX)/2;
freq = (1*Sampling_rate);
t=[ST:1:(LENX+ST-1)]*(step_dist/Sampling_rate)*tune_X;
frequency=Sampling_rate*((1:IMGY)/(IMGY*4))*tune_Y;
imagesc(t,frequency,y');
ylabel('Frequency');
title('Short Time Fourier Transform')
axis('xy')
Fre_min=Fre_min*tune_Y;
Fre_max=Fre_max*tune_Y;
for ckj=1:1:IMGY
    if Fre_max < frequency(ckj)
        break
    end
end
[CMAX,CMIN,CMEA] = CValue (y(:,1:ckj),0.1,0.9);
caxis([ CMIN CMAX])            

axis([ tt(1) tt(l) Fre_min Fre_max])
ttt=t;
f=frequency;
z=y';
end


if (display_mode == 2 | display_mode == 4 ) & mean2(wps) ~= 0 & Show_CK == 3
    if display_mode == 2
        figure
    else
        subplot(4,4,9:12)
    end

t=t*tune_X;
frequency=frequency*tune_Y;
surfc(t, frequency ,wps );
shading interp;
%axis([0, t(l) frequency(j1+1) Fre_max])
ylabel('Frequency');
title('Wavelet Transform')
Fre_min=Fre_min*tune_Y;
Fre_max=Fre_max*tune_Y;
axis([ tt(1) tt(l) Fre_min Fre_max])
[CMAX,CMIN,CMEA] = CValue (wps,0.1,0.7);
caxis([ CMIN CMAX])            

%caxis([ 10^mean2(log(wps'))-0.005*10^std2(log(wps')) 10^mean2(log(wps'))+0.005*10^std2(log(wps'))])            
ttt=t;
f=frequency;
z=wps;
end



if (display_mode == 3 | display_mode == 4 ) & mean2(Z_Ori) ~= 0 & Show_CK == 4

    if display_mode == 3
        figure
    else
        subplot(4,4,13:16)
    end

%IMF_Amp_Matrix=Z_Ori;
[X_len,Y_len]=size(Z_Ori);
Y_max=Fre_max;%max(max(Y_Ori)); %定義Y軸的最大值
[XI,YI]=meshgrid(1:1:X_len,Fre_min:resolution:Fre_max);

[X_Olen,Y_Olen]=size(XI');
ZI(1:X_Olen,1:Y_Olen)=0;,

for xx=1:1:X_len
    for yy=1:1:Y_len
        if Y_Ori(xx,yy) >= 0 & Y_Ori (xx,yy) <= Y_max
               Y_nor=fix((Y_Ori(xx,yy)-Fre_min)/resolution)+1;
               if ZI(xx,Y_nor) <= Z_Ori(xx,yy)
               ZI(xx,Y_nor)=Z_Ori(xx,yy);
               end
               
        end
    end
end

t=t*tune_X;
frequency=frequency*tune_Y;
imagesc(t, frequency ,ZI');
axis('xy')
[CMAX,CMIN,CMEA] = CValue (Z_Ori,0.2,0.8);
caxis([ CMIN CMAX])            
%caxis([ mean2(Z_Ori)-0.4*std2(Z_Ori) mean2(Z_Ori)+0.25*std2(Z_Ori)])            
xlabel('\bfYear','Fontsize',16);
ylabel('\bfPeriod','Fontsize',16);
title('\bfHilbert-Huang Transform for Hualien 1922-2008','Fontsize',16)
set(gca,'YTick',[1/365/11 1/365/5 1/365/3 1/365/2 1/365 1/365/0.7 1/365/0.5])
set(gca,'YTickLabel',{'11 years';'5 years';'3 years';'2 years';'One Year';'0.7 year';'Half year'},'Fontsize',16)
set(gca,'XTick',[0 8 18 28 38 48 58 68 78 86])
set(gca,'XTickLabel',{'1922';'1930';'1940';'1950';'1960';'1970';'1980';'1990';'2000';'2008'},'Fontsize',16)
Fre_min=Fre_min*tune_Y;
Fre_max=Fre_max*tune_Y;
caxis([0 2])
colorbar('YTickLabel',{'0.0','0.2','0.4','0.6','0.8','1','1.2','1.4','1.6','1.8','2.0'},'FontSize',16)

%axis([ tt(1) tt(l) Fre_min Fre_max])
ttt=t;
f=frequency;
z=ZI';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [CMAX,CMIN,CMEA] = CValue (x,C20,C80)
nT=0;
[sx,sy]=size(x);
if sy > sx, x=x';, end
[sx,sy]=size(x);
Xmax=nanmax(nanmax(x));
Xmin=nanmin(nanmin(x));
Xint=Xmin:(Xmax-Xmin)/1000:Xmax;
for yy=1:1:sy
[n,nout]=hist(x(:,yy),Xint);
nT=nT+n;
end
nT20=sum(nT)*C20;
nT80=sum(nT)*C80;
nT50=sum(nT)*0.5;
CV_s=0;
if length(nT) ~= 0
for nT_num=1:1:1000
CV_m=CV_s+nT(nT_num);
   if CV_s <= nT80 & nT80 < CV_m
       CMAX=Xint(nT_num+1)-(((CV_m-nT80)/nT(nT_num))*((Xmax-Xmin)/1000));
   end
   if CV_s <= nT50 & nT50 < CV_m
       CMEA=Xint(nT_num+1)-(((CV_m-nT50)/nT(nT_num))*((Xmax-Xmin)/1000));
   end
   if CV_s <= nT20 & nT20 < CV_m
       CMIN=Xint(nT_num+1)-(((CV_m-nT20)/nT(nT_num))*((Xmax-Xmin)/1000));
   end
CV_s=CV_m;
end
else
CMAX = Xmax;
CMIN = Xmin;
CMEA = (Xmax+Xmin)/2;
end