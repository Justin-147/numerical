%AA=load('010845.11.pos');
clear
DOY=365;
Rday=datenum(2007,1,1);
Data_path='E:\日本GPS資料\2007';
Out_path='E:\日本GPS資料\temp2007';
KKK=dir(Data_path);


fidst=fopen(strcat(Out_path,'\','stinfo.mat'));
if fidst > 0
    load(strcat(Out_path,'\','stinfo.mat'));
    fclose(fidst);
end
    

for jj=984:1:length(KKK)  
    if  jj~=99999
    jj
filenamet=KKK(jj).name;
filename=strcat(Data_path,'\',filenamet);

fid=fopen(filename);
if fid >= 0
[YYYY,MM,DD,Skip1,Skip2,Skip3,Skip4,Skip5,Skip6,GPSD1,GPSD2,GPSD3]=textread(filename,'%5s %3s %3s %3s %3s %3s %18s %18s %18s %18s %18s %18s','delimiter','\n','whitespace','','headerlines',20);
for Lrun=1:1:length(GPSD1)-1
    
Ctemp=(YYYY{Lrun});
nYYYY(Lrun,1)=str2num(Ctemp);
Ctemp=(MM{Lrun});
nMM(Lrun,1)=str2num(Ctemp);
Ctemp=(DD{Lrun});
nDD(Lrun,1)=str2num(Ctemp);
Ctemp=(GPSD1{Lrun});
nData1(Lrun,1)=str2num(Ctemp);
Ctemp=(GPSD2{Lrun});
nData2(Lrun,1)=str2num(Ctemp);
Ctemp=(GPSD3{Lrun});
nData3(Lrun,1)=str2num(Ctemp);
end
Timeindex=datenum(nYYYY,nMM,nDD)-Rday+1;
C10(Timeindex,1)=Timeindex;
C10(Timeindex,4)=nData1;
C10(Timeindex,5)=nData2;
C10(Timeindex,6)=nData3;
fclose(fid);
end

inddd=0;
if ((Timeindex(1) ~= 1) |  (Timeindex(end) ~= DOY) | (length(YYYY) <= DOY*.95))
inddd=1;    
end
clear Skip1 Skip2 Skip3 Skip4 Skip5 Skip6 nYYYY nMM nDD nData1 nData2 nData3 MM DD Data1 Data2 Data3

if inddd == 0
    
filename15=filenamet(1:end-6);
filename17=filenamet(1:end-4);  
filename1013=filenamet(end-3:end);

lo(jj-2,1)=median(C10(:,4));
lo(jj-2,2)=median(C10(:,5));
lo(jj-2,3)=str2num(filename15);

%----------------------------------
CCT=C10(:,4);
QQss=find(abs(CCT) > 0);
C1011(1:DOY,4)=interp1(QQss,CCT(QQss),1:1:DOY);
%%%%%%%%%%%%%%%%%%%%%%%%%%%  轉換至地阜ヱ直座標 (135,35)
Eavr=6371008.7714;
La_surface=(2*pi*Eavr/360)*(C1011(:,4)-35);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

QQs=La_surface-La_surface(1);
QQlb=quantile(QQs,.01);
QQub=quantile(QQs,.99);
QQsss=find(QQs >= QQlb & QQs <= QQub);
QQsssl=length(QQsss);
if QQsss(1) ~= 1, QQsss(2:QQsssl+1)=QQsss; QQsss(1)=1; end
QQsssl=length(QQsss);
if QQsss(end) ~= DOY, QQsss(QQsssl+1)=DOY; end
QQs=interp1(QQsss,QQs(QQsss),1:1:DOY);

[FFT_t,FFT_f,FFT_z,WT_t,WT_f,WT_z,HHT_t,HHT_f,HHT_z,IMF_Matrix,IMF_Amp_Matrix,IMF_Fre_Matrix] = TFMain(QQs');        
[kx,ky]=size(IMF_Fre_Matrix);
JJ=[];
for j=1:1:kx
Fre_sel=find(IMF_Fre_Matrix(j,:) > 1/150/86400 & IMF_Fre_Matrix(j,:) < 1/20/86400);
NS_ll(j)=(sum(IMF_Matrix(j,Fre_sel)));
KK(j)=(sum(IMF_Amp_Matrix(j,Fre_sel)));
end

NS_Amp=IMF_Amp_Matrix;
NS_Fre=IMF_Fre_Matrix;
NS=IMF_Matrix;

clear IMF_Fre_Matrix IMF_Amp_Matrix IMF_Matrix
CCT=C10(:,5);
QQss=find(abs(CCT) > 0);
C1011(1:DOY,5)=interp1(QQss,CCT(QQss),1:1:DOY);
%%%%%%%%%%%%%%%%%%%%%%%%%%%  轉換至地阜ヱ直座標 (135,35)
Eavr=6371008.7714;
Elrad=6378137;
EEE2=0.00669438;
RR=Elrad./(sqrt(1-(EEE2*(sin(C1011(:,4)/180*pi).^2))));
RR2=RR.*cos(C1011(:,4)/180*pi);
Lo_surface=2*pi*RR2/360.*(C1011(:,5)-137);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

QQs=Lo_surface-Lo_surface(1);

QQlb=quantile(QQs,.01);
QQub=quantile(QQs,.99);
QQsss=find(QQs >= QQlb & QQs <= QQub);
QQsssl=length(QQsss);
if QQsss(1) ~= 1, QQsss(2:QQsssl+1)=QQsss; QQsss(1)=1; end
QQsssl=length(QQsss);
if QQsss(end) ~= DOY, QQsss(QQsssl+1)=DOY; end
QQs=interp1(QQsss,QQs(QQsss),1:1:DOY);

[FFT_t,FFT_f,FFT_z,WT_t,WT_f,WT_z,HHT_t,HHT_f,HHT_z,IMF_Matrix,IMF_Amp_Matrix,IMF_Fre_Matrix] = TFMain(QQs');        
[kx,ky]=size(IMF_Fre_Matrix);
JJ=[];
for j=1:1:kx
Fre_sel=find(IMF_Fre_Matrix(j,:) > 1/150/86400 & IMF_Fre_Matrix(j,:) < 1/20/86400);
EW_ll(j)=(sum(IMF_Matrix(j,Fre_sel)));
CC(j)=(sum(IMF_Amp_Matrix(j,Fre_sel)));
end
EW_Amp=IMF_Amp_Matrix;
EW_Fre=IMF_Fre_Matrix;
EW=IMF_Matrix;


% 計?vertical----------------------------------------------------------------------------------
CCT=C10(:,6);
QQss=find(abs(CCT) > 0);
C1011(1:DOY,6)=interp1(QQss,CCT(QQss),1:1:DOY);
QQs=C1011(:,6)-C1011(1,6);
QQlb=quantile(QQs,.01);
QQub=quantile(QQs,.99);
QQsss=find(QQs >= QQlb & QQs <= QQub);
QQsssl=length(QQsss);
if QQsss(1) ~= 1, QQsss(2:QQsssl+1)=QQsss; QQsss(1)=1; end
QQsssl=length(QQsss);
if QQsss(end) ~= DOY, QQsss(QQsssl+1)=DOY; end
QQs=interp1(QQsss,QQs(QQsss),1:1:DOY);
[FFT_t,FFT_f,FFT_z,WT_t,WT_f,WT_z,HHT_t,HHT_f,HHT_z,IMF_Matrix,IMF_Amp_Matrix,IMF_Fre_Matrix] = TFMain(QQs');        
[kx,ky]=size(IMF_Fre_Matrix);
JJ=[];
for j=1:1:kx
Fre_sel=find(IMF_Fre_Matrix(j,:) > 1/150/86400 & IMF_Fre_Matrix(j,:) < 1/20/86400);
ZZ_ll(j)=(sum(IMF_Matrix(j,Fre_sel)));
ZZ_Amp(j)=(sum(IMF_Amp_Matrix(j,Fre_sel)));
end

Z_Amp=IMF_Amp_Matrix;
Z_Fre=IMF_Fre_Matrix;
Z=IMF_Matrix;
%------------------------------------------------------------------------ 





for ll=1:1:length(NS_ll)
    if EW_ll(ll) ~= 0 &  NS_ll(ll) ~= 0
AA(ll)=atan2(EW_ll(ll),NS_ll(ll));
AAQ(ll)=sqrt(NS_ll(ll)^2+EW_ll(ll)^2);
    else
AA(ll)=nan;
AAQ(ll)=nan;
    end
end
AA_s=find(AA < 0);
AA(AA_s)=2*pi+AA(AA_s);
AA=AA/pi*180;

    AZI(1:DOY,jj-2)=AA;
    SDF(1:DOY,jj-2)=AAQ;

savefile=strcat(Out_path,'\',filename15,'mat');
save(savefile,'NS_Amp','NS_Fre','NS','EW_Amp','EW_Fre','EW','AA','NS_ll','EW_ll','Z_Amp','Z_Fre','Z','ZZ_ll')
clear AA AAQ GPS C T  IMF_Fre_Matrix IMF_Amp_Matrix IMF_Matrix  EW_Amp EW_Fre EW NS_Amp NS_Fre NS KK CC EW_ll NS_ll P Cdd Z_Amp Z_Fre Z ZZ_ll

close(1)
close(2)
close(3)
clear C10 C11 C1011
end
    end
end
save(strcat(Out_path,'\','stinfo.mat'),'lo')
