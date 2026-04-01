%EEMD(Ensemble Empirical Mode Decomposition)
%function [E,C]=EEMD(i_n,e_n,h_n_e,h_n_c,c_n,nval);
function [E]=EEMD(i_n,e_n,h_n_e,nval);
%                                                syy
%============================>>>>  input  <<<<=============================
%i_n data
%e_n 加幾次noise
%h_n_e ensemble 部分 Sifting的次數
%h_n_c ensemble 後剩下的東西做EMD Sifting的次數
%nval white noise振幅大小
%c_n 猜看看得到幾個IMF

yi=i_n;
[I,J]=size(yi);
if I > J;
    l_yi=I;
    yi=yi';
else
    l_yi=J;
    yi=yi;
end

xi=1:l_yi;

%=========================EEEEE============================================
%=========================noise============================================
    
    %seed random number generator
    rand('state',sum(100*clock));
    %calculate rms
    %rms = norm(yi)/sqrt(l_yi);
    
    m_a=median(yi);
    
    
    %calculate noise scale
    scale=nval/max(abs((yi-m_a)));
%    scale = median(abs((yi-m_a)*nval/1000));
    E=zeros(e_n,l_yi);
%    C=zeros(c_n,l_yi);
for j=1:e_n;
    
    noise = (rand(1,l_yi)-0.5000)*2*scale;
%=========================================================================
    
    yi=yi+noise;
    [imf_e]=sifting(xi,yi,h_n_e);
    if yi == imf_e;break;end
    yi=yi-imf_e;
    E(j,:)=imf_e;
%    eval([  'disp(''!!!!!!!!!!!!!!!!!!! E' num2str(j) ''');' ]);
end

E=[E(1:j,:);yi-noise];
%disp('!!!!!!!!!!!!!!!!!!  E over!');
%=========================================================================



%=========================CCCCC============================================
%if yi ~= imf_e;
%    for k=1:c_n;
%        [imf_c]=sifting(xi,yi,h_n_c);
%        if yi == imf_c;break;end
%        yi=yi-imf_c;
%        C(k,:)=imf_c;
%        eval([  'disp(''!!!!!!!!!!!!!!!!!!!  C' num2str(k) ''');' ]);
%    end
%    
%    C=[C(1:k,:);yi];
%    disp('!!!!!!!!!!!!!!!!!!  C over!');
%
%else
%    C=[];
%    disp('!!!!!!!!!!!!!!!!!!  C over!');
%end
%==========================================================================


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sifting 每執行完一次就得到一個IMF
function [imf]=sifting(xi,yi,h_n);
% input  xi = 時間或位置
%        yi = 數值大小
%        h_n = sifting的次數
%
% output c = imf*1
l_yi=length(yi);
for i=1:h_n;
    [ex_min_x,ex_min_y]=locomi(yi);
    [ex_max_x,ex_max_y]=locoma(yi);
    if length(ex_max_x) < 2 | length(ex_min_x) < 2 ;break;end
    [max_x, max_y, min_x, min_y] = mypredict(xi, yi, ex_max_x, ex_max_y, ex_min_x, ex_min_y);
%    [min_x,max_x,min_y,max_y] = boundary_conditions(ex_min_x,ex_max_x,xi,yi,yi);

    [Eyi_u,Eyi_d,tu,td]=envelopef(max_y,min_y,max_x,min_x,l_yi);
    m_y=(Eyi_u+Eyi_d)/2;
    if abs(m_y) <= 10^-6;break;end    
    yi=yi-m_y;
%    eval([  'disp(''sifting' num2str(i) ''');' ]);
end
imf=yi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Find the maginitude and locations of the maximun and minmun in the data
function [x_u,y_u]=locoma(data);
% input  data = data matrix [1*n]
%
% output f_au = 極大值的值
%        f_ad = 極小值的值
%        t_au = 極大值在原矩陣的位置
%        t_ad = 極小值在原矩陣的位置
y=data;
l_y=length(y);

y_u=zeros(round(l_y/2),1);
x_u=zeros(round(l_y/2),1);

countu=0;

for i=2:l_y-1;
    if y(i-1) < y(i) & y(i) > y(i+1);
        countu=countu+1;
        y_u(countu,1)=y(i);
        x_u(countu,1)=i;
    else if y(i-1) < y(i) & y(i) == y(i+1);
            countu=countu+1;
            y_u(countu,1)=y(i);
            x_u(countu,1)=i;
        end
    end
end

y_u=y_u(1:countu)';
x_u=x_u(1:countu)';

l_u=length(y_u);

%eval([' disp('' u = ' num2str(l_u) '''); ']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Find the maginitude and locations of the maximun and minmun in the data
function [x_d,y_d]=locomi(data);
% input  data = data matrix [1*n]
%
% output f_au = 極大值的值
%        f_ad = 極小值的值
%        t_au = 極大值在原矩陣的位置
%        t_ad = 極小值在原矩陣的位置
y=data;
l_y=length(y);

y_d=zeros(round(l_y/2),1);
x_d=zeros(round(l_y/2),1);

countd=0;

for i=2:l_y-1;
    if y(i-1) > y(i) & y(i) < y(i+1);
        countd=countd+1;
        y_d(countd,1)=y(i);
        x_d(countd,1)=i;
    else if y(i-1) > y(i) & y(i) == y(i+1);
            countd=countd+1;
            y_d(countd,1)=y(i);
            x_d(countd,1)=i;
        end
    end
end

y_d=y_d(1:countd)';
x_d=x_d(1:countd)';

l_d=length(y_d);

%eval([' disp('' d = ' num2str(l_d) '''); ']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [max_x, max_y, min_x, min_y] = mypredict(data_x, data_y, ex_max_x, ex_max_y, ex_min_x, ex_min_y);

l_ma=length(ex_max_x);
l_mi=length(ex_min_x);
std_max_y=std(ex_max_y);
std_min_y=std(ex_min_y);

%=============================max==========================================
if l_ma >=3;
    %============end============================================
    m_maxEX=round(mean(diff(ex_max_x(end-2:end))));
    p_maxEX1=ex_max_x(end)+ceil((data_x(end)-ex_max_x(end))/m_maxEX)*m_maxEX;
    p_maxEX2=p_maxEX1+m_maxEX;
    p_maxEX=[p_maxEX1,p_maxEX2];
    
    m_maxEY=mean(ex_max_y(end-2:end));
    if data_y(end) > data_y(end-1);
        if m_maxEY > data_y(end);
            p_maxEY = [m_maxEY,m_maxEY];
        else
            m_maxEY=mean(abs(diff(ex_max_y(end-2:end))));
            p_maxEY=[m_maxEY,m_maxEY]/2+data_y(end);
        end
    else
        p_maxEY = [m_maxEY,m_maxEY];
    end
    
    %============begin==========================================
    m_maxBX=round(mean(diff(ex_max_x(1:3))));
    p_maxBX1=(ex_max_x(1)-data_x(1))-ceil((ex_max_x(1)-data_x(1))/m_maxBX)*m_maxBX;
    p_maxBX2=p_maxBX1-m_maxBX;
    p_maxBX=[p_maxBX2,p_maxBX1];
    
    m_maxBY=mean(ex_max_y(1:3));
    if data_y(1) > data_y(2);
        if m_maxBY > data_y(1);
            p_maxBY = [m_maxBY,m_maxBY];
        else
            m_maxBY=mean(abs(diff(ex_max_y(1:3))));
            p_maxBY=[m_maxBY,m_maxBY]/2+data_y(1);
        end
    else
        p_maxBY = [m_maxBY,m_maxBY];
    end
else
    %============end============================================
    p_maxEX=[data_x(end)+2,data_x(end)+4];
    p_maxEY=[ex_max_y(end),ex_max_y(end)];
    %============begin==========================================
    p_maxBX=[data_x(1)-4,data_x(1)-2];
    p_maxBY=[ex_max_y(1),ex_max_y(1)];
end


%=============================min==========================================
if l_mi >=3;
    %============end============================================
    m_minEX=round(mean(diff(ex_min_x(end-2:end))));
    p_minEX1=ex_min_x(end)+ceil((data_x(end)-ex_min_x(end))/m_minEX)*m_minEX;
    p_minEX2=p_minEX1+m_minEX;
    p_minEX=[p_minEX1,p_minEX2];
    
    m_minEY=mean(ex_min_y(end-2:end));
    if data_y(end) < data_y(end-1);
        if m_minEY < data_y(end);
            p_minEY = [m_minEY,m_minEY];
        else
            m_minEY=mean(abs(diff(ex_min_y(end-2:end))));
            p_minEY=[m_minEY,m_minEY]/2+data_y(end);
        end
    else
        p_minEY = [m_minEY,m_minEY];
    end
    
    %============begin==========================================
    m_minBX=round(mean(diff(ex_min_x(1:3))));
    p_minBX1=(ex_min_x(1)-data_x(1))-ceil((ex_min_x(1)-data_x(1))/m_minBX)*m_minBX;
    p_minBX2=p_minBX1-m_minBX;
    p_minBX=[p_minBX2,p_minBX1];
    
    m_minBY=mean(ex_min_y(1:3));
    if data_y(1) < data_y(2);
        if m_minBY < data_y(1);
            p_minBY = [m_minBY,m_minBY];
        else
            m_minBY=mean(abs(diff(ex_min_y(1:3))));
            p_minBY=[m_minBY,m_minBY]/2+data_y(1);
        end
    else
        p_minBY = [m_minBY,m_minBY];
    end
else
    %============end============================================
    p_minEX=[data_x(end)+2,data_x(end)+4];
    p_minEY=[ex_min_y(end),ex_min_y(end)];
    %============begin==========================================
    p_minBX=[data_x(1)-4,data_x(1)-2];
    p_minBY=[ex_min_y(1),ex_min_y(1)];
end

max_x=[p_maxBX,ex_max_x,p_maxEX];
max_y=[p_maxBY,ex_max_y,p_maxEY];
min_x=[p_minBX,ex_min_x,p_minEX];
min_y=[p_minBY,ex_min_y,p_minEY];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Envelope
function [Eyi_u,Eyi_d,tu,td]=envelopef(f_au,f_ad,t_au,t_ad,l_yi);
% input  x = 時間或位置
%        y = 數值大小
%        f_au = 極大值的值
%        f_ad = 極小值的值
%        t_au = 極大值在原矩陣的位置
%        t_ad = 極小值在原矩陣的位置
%
% output Eyi_u = 上包羅線
%        Eyi_d = 下包羅線


%上包羅線

x=t_au;
y=f_au;
l=length(x);
tu=x(1):x(l);
Eyi_u=spline(x,y,tu);
c_maxbi=find(tu==1);
Eyi_u=Eyi_u(c_maxbi:c_maxbi+l_yi-1);
tu=tu(c_maxbi:c_maxbi+l_yi-1);
%c_maxb=1-x(1);
%Eyi_u=Eyi_u(c_maxb+1:c_maxb+l_yi);
%tu=tu(c_maxb+1:c_maxb+l_yi);
%clear x y l c_bi
%下包羅線

x=t_ad;
y=f_ad;
l=length(x);
td=x(1):x(l);
Eyi_d=spline(x,y,td);
c_minbi=find(td==1);
Eyi_d=Eyi_d(c_minbi:c_minbi+l_yi-1);
td=td(c_minbi:c_minbi+l_yi-1);
%c_minb=1-x(1);
%Eyi_d=Eyi_d(c_minb+1:c_minb+l_yi);
%td=td(c_minb+1:c_minb+l_yi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% defines new extrema points to extend the interpolations at the edges of the
% signal (mainly mirror symmetry)
function [tmin,tmax,zmin,zmax] = boundary_conditions(indmin,indmax,t,x,z)
	nbsym=2;
	lx = length(x);
	
	if (length(indmin) + length(indmax) < 3)
		error('not enough extrema')
	end

    % boundary conditions for interpolations :

	if indmax(1) < indmin(1)
    	if x(1) > x(indmin(1))
			lmax = fliplr(indmax(2:min(end,nbsym+1)));
			lmin = fliplr(indmin(1:min(end,nbsym)));
			lsym = indmax(1);
		else
			lmax = fliplr(indmax(1:min(end,nbsym)));
			lmin = [fliplr(indmin(1:min(end,nbsym-1))),1];
			lsym = 1;
		end
	else

		if x(1) < x(indmax(1))
			lmax = fliplr(indmax(1:min(end,nbsym)));
			lmin = fliplr(indmin(2:min(end,nbsym+1)));
			lsym = indmin(1);
		else
			lmax = [fliplr(indmax(1:min(end,nbsym-1))),1];
			lmin = fliplr(indmin(1:min(end,nbsym)));
			lsym = 1;
		end
	end
    
	if indmax(end) < indmin(end)
		if x(end) < x(indmax(end))
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
			rmin = fliplr(indmin(max(end-nbsym,1):end-1));
			rsym = indmin(end);
		else
			rmax = [lx,fliplr(indmax(max(end-nbsym+2,1):end))];
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
			rsym = lx;
		end
	else
		if x(end) > x(indmin(end))
			rmax = fliplr(indmax(max(end-nbsym,1):end-1));
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
			rsym = indmax(end);
		else
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
			rmin = [lx,fliplr(indmin(max(end-nbsym+2,1):end))];
			rsym = lx;
		end
	end
    
	tlmin = 2*t(lsym)-t(lmin);
	tlmax = 2*t(lsym)-t(lmax);
	trmin = 2*t(rsym)-t(rmin);
	trmax = 2*t(rsym)-t(rmax);
    
	% in case symmetrized parts do not extend enough
	if tlmin(1) > t(1) || tlmax(1) > t(1)
		if lsym == indmax(1)
			lmax = fliplr(indmax(1:min(end,nbsym)));
		else
			lmin = fliplr(indmin(1:min(end,nbsym)));
		end
		if lsym == 1
			error('bug')
		end
		lsym = 1;
		tlmin = 2*t(lsym)-t(lmin);
		tlmax = 2*t(lsym)-t(lmax);
	end   
    
	if trmin(end) < t(lx) || trmax(end) < t(lx)
		if rsym == indmax(end)
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
		else
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
		end
	if rsym == lx
		error('bug')
	end
		rsym = lx;
		trmin = 2*t(rsym)-t(rmin);
		trmax = 2*t(rsym)-t(rmax);
	end 
          
	zlmax =z(lmax); 
	zlmin =z(lmin);
	zrmax =z(rmax); 
	zrmin =z(rmin);
     
	tmin = [tlmin t(indmin) trmin];
	tmax = [tlmax t(indmax) trmax];
	zmin = [zlmin z(indmin) zrmin];
	zmax = [zlmax z(indmax) zrmax];

