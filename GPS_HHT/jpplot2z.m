clear
load('tohokumap.mat')
load('plotdata.mat')
load('colortohoku.mat')
for jj=1:1:434
TEMP=kak_za(:,:,jj);
ZISO=find(ZI<=0);%100000);
ZI2(101:401,101:401)=TEMP;
clear TEMP
ZI2(401:501,:)=nan;
ZI2(1:101,:)=nan;
ZI2(:,401:501)=nan;
ZI2(:,1:101)=nan;
ZI2(301:401,102:250)=nan;
ZI2(ZISO)=nan;%ZI(ZISO);
surf(XI,YI,ZI,ZI2)
shading flat
view(0,90)
hold on
contour(XI,YI,ZI,[0,0],'-','linewidth',2,'color','k');
%colormap(cmap)
caxis([-0.006  0.006])
set(gca,'yTick',25:5:50,'yTickLabel',{'25' '30' '35' '40' '45' '50'},'FontSize',14)
set(gca,'xTick',125:5:150,'xTickLabel',{'125' '130' '135' '140' '145' '150'},'FontSize',14)
xlabel('\bfLongitude','fontsize',14)
ylabel('\bfLatitude','fontsize',14)
title(strcat('\bfEQD',num2str(-13+(jj-422))))
box on
colorbar
%colorbar('ytick',[-9000 -6000 -3000 0 200 400 1000],'yticklabel','-9000|-6000|-3000|0|0.02(50¢X)|0.04(25¢X)|0.1(10¢X)','fontsize',8);
pstx=145;
psty=30;
ll=0.005*500;
llaa=90;
penx=pstx+ll*sin(llaa/180*pi);
peny=psty+ll*cos(llaa/180*pi);
%plot([pstx penx penx+0.2*sin((llaa-150)/180*pi) penx penx+0.2*sin((llaa+150)/180*pi)],[psty peny peny+0.2*cos((llaa-150)/180*pi) peny peny+0.2*cos((llaa+150)/180*pi)],'-k','linewidth',1)
%text(145.3,29.5,'0.5mm','fontsize',10)
plot(142.836,38.424,'pentagram','markersize',10,'color','r','markerfacecolor','r')
text(143.2,38.424,'M=7.3','color','r')
plot(142.369,38.322,'pentagram','markersize',16,'color','r','markerfacecolor','r')
text(141.769,37.522,'M=9','color','r')
text(156,30,'\bfResidual vertical displacement (m)','color','k','rotation',90,'FontSize',14)
%text(156,47,'\bfGPS Index','color','k','rotation',90)
    

filename=strcat('ZEQD2',num2str(jj,'%.3d'));
%set(gcf,'PaperPosition', [0 0 36 24]); %Position plot at left hand corner with width 5 and 
saveas(gcf,filename,'png')
close(1)
end
