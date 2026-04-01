    %AA=load('010845.11.pos');
clear
%KKK=dir('C:\Users\nononoCHChen\Desktop\gps data\2011');
%for jj=3:1:length(KKK)  %st~=66
%    jj
%filenamet=KKK(jj).name;
%filename=strcat('C:\Users\nononoCHChen\Desktop\gps
%data\2011\',filenamet);
%fid=fopen(filename);
%if fid >= 0   
%data=textread(filename,'%s','delimiter','\n','whitespace','');
%L=length(data);
%Lr=L;
%Ct=dlmread(filename,'\s',strcat('E20..J',num2str(L-2)));
%C11=Ct(2:end,1:3);
%fclose(fid);
%end
%lo(jj-2,1)=median(Ct(:,4));
%lo(jj-2,2)=median(Ct(:,5));
%lo(jj-2,3)=str2num(filenamet(1:6));
%end
Out_path='C:\Users\nonono\Desktop\GPS';
load(strcat(Out_path,'\','stinfo.mat'));

smallyc=0.5;
largeyc=2;

Anom_la=44;
Anom_lo=142;
DOY=365;

%load coast;plot(long,lat,'k-','markersize',1);axis([125 155 25 50]);
%hold on
%plot(lo(1:end,2),lo(1:end,1),'.')
int=0.05;
slint=0.5;
lonnn=0;
for lon=130:int:145
    lonnn=lonnn+1;
    disp(lonnn/((145-130)/int+1)*100)
    lattt=0;
    for lat=30:int:45
        lattt=lattt+1;
        STs=find(lo(:,1) >= lat-slint & lo(:,1) <= lat+slint & lo(:,2) >= lon-slint & lo(:,2) <= lon+slint);
        if length(STs)>=3
            Ang=[]; STF=[];
            analen=0;
           for ll=1:1:length(STs)
               filename=(strcat(Out_path,'\',num2str(lo(STs(ll),3),'%.6d'),'.mat'));
               fid=fopen(filename);
               if fid > 0
                   load(filename);
                   if length(EW_ll) == DOY
               analen=analen+1;
               load(filename);
               AA=atan2(EW_ll,NS_ll);
               AA_s=find(AA < 0);
               AA(AA_s)=2*pi+AA(AA_s);
               AA=AA/pi*180;               
               AAs=sqrt(EW_ll.^2+NS_ll.^2);
               AAz=ZZ_ll;
               ZAmp(:,analen)=AAz;
               Ang(:,analen)=AA;
               STF(:,analen)=AAs;
                   end
               fclose(fid);
               end
           end 
               for day=1:1:DOY
               DDs=0; stnn=0;
               for j1=1:1:analen-1
               for j2=j1+1:1:analen
                  if Ang(day,j1)>=0 & Ang(day,j2)>=0
                  stnn=stnn+1;
                  DD=abs(Ang(day,j1)-Ang(day,j2));
                  if DD > 180, DD=360-DD;, end
                  DDs=DDs+(DD);
                  end
               end
               end
               if length(Ang) ~= 0
               kaka(lattt,lonnn,day)=nanmedian(Ang(day,:));  % kaka GPS 方向
               kaks(lattt,lonnn,day)=nanmean(STF(day,:));   % kaks GPS 大小
               kak_za(lattt,lonnn,day)=nanmedian(ZAmp(day,:));  % kakz GPS 垂直量
               MMAP(lattt,lonnn)=length(STs);   % kaks GPS 大小
               else
               kaka(lattt,lonnn,day)=nan;  % kaka GPS 方向
               kaks(lattt,lonnn,day)=nan;   % kaks GPS 大小
               kak_za(lattt,lonnn,day)=nan;  % kakz GPS 垂直量
               end                           
               kak(lattt,lonnn,day)=1/(DDs/stnn);            % 角度差異量
               end
        else
        kak(lattt,lonnn,1:DOY)=nan;            % 角度差異量        
        end
    end
end


for lon=Anom_lo:int:Anom_lo
    lonnn=lonnn+1;
    lattt=0;
    for lat=Anom_la:int:Anom_la
        lattt=lattt+1;
        STs=find(lo(:,1) >= lat-largeyc & lo(:,1) <= lat+largeyc & lo(:,2) >= lon-largeyc & lo(:,2) <= lon+largeyc);
        if length(STs)>=3
            Ang=[]; STF=[];
            analen=0;
           for ll=1:1:length(STs)
               filename=(strcat(num2str(lo(STs(ll),3),'%.6d'),'.mat'));
               fid=fopen(filename);
               if fid > 0
                   load(filename);
                   if length(EW_ll) == DOY
               analen=analen+1;
               load(filename);
               AA=atan2(EW_ll,NS_ll);
               AA_s=find(AA < 0);
               AA(AA_s)=2*pi+AA(AA_s);
               AA=AA/pi*180;               
               AAs=sqrt(EW_ll.^2+NS_ll.^2);
               AAz=ZZ_ll;
               ZAmp(:,analen)=AAz;
               Ang(:,analen)=AA;
               STF(:,analen)=AAs;
               fclose(fid);
                   end
               end
           end 
               for day=1:1:DOY
               DDs=0; stnn=0;
               for j1=1:1:analen-1
               for j2=j1+1:1:analen
                  if Ang(day,j1)>=0 & Ang(day,j2)>=0
                  stnn=stnn+1;
                  DD=abs(Ang(day,j1)-Ang(day,j2));
                  if DD > 180, DD=360-DD;, end
                  DDs=DDs+(DD);
                  end
               end
               end
               kak_largeyc(day)=1/(DDs/stnn);            % 角度差異量
               end
        else
        kak_largeyc(1:DOY)=nan;            % 角度差異量        
        end
    end
end

for lon=Anom_lo:int:Anom_lo
    lonnn=lonnn+1;
    lattt=0;
    for lat=Anom_la:int:Anom_la
        lattt=lattt+1;
        STs=find(lo(:,1) >= lat-smallyc & lo(:,1) <= lat+smallyc & lo(:,2) >= lon-smallyc & lo(:,2) <= lon+smallyc);
        if length(STs)>=3
            Ang=[]; STF=[];
            analen=0;
           for ll=1:1:length(STs)
               filename=(strcat(num2str(lo(STs(ll),3),'%.6d'),'.mat'));
               fid=fopen(filename);
               if fid > 0
                   load(filename);
                   if length(EW_ll) == DOY
               analen=analen+1;
               load(filename);
               AA=atan2(EW_ll,NS_ll);
               AA_s=find(AA < 0);
               AA(AA_s)=2*pi+AA(AA_s);
               AA=AA/pi*180;               
               AAs=sqrt(EW_ll.^2+NS_ll.^2);
               AAz=ZZ_ll;
               ZAmp(:,analen)=AAz;
               Ang(:,analen)=AA;
               STF(:,analen)=AAs;
               fclose(fid);
                   end
               end
           end 
               for day=1:1:DOY
               DDs=0; stnn=0;
               for j1=1:1:analen-1
               for j2=j1+1:1:analen
                  if Ang(day,j1)>=0 & Ang(day,j2)>=0
                  stnn=stnn+1;
                  DD=abs(Ang(day,j1)-Ang(day,j2));
                  if DD > 180, DD=360-DD;, end
                  DDs=DDs+(DD);
                  end
               end
               end
               kak_smallyc(day)=1/(DDs/stnn);            % 角度差異量
               end
        else
        kak_smallyc(1:DOY)=nan;            % 角度差異量        
        end
    end
end

save(strcat(Out_path,'\','plotdata.mat'),'kak_smallyc','kak_largeyc','kak','kaka','kaks','kak_za','int','MMAP')
    
               
               
               
               
                   
               
               
               
               
            