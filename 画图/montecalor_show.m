
function montecalor_show(data_in_3_demention,int,colormatinput,titlename,plottype,textlable)
% clf
% set(gcf, 'position', [0 0 900 800]);
if nargin==6
x=-200+2.5: 5 :200+2.5;
y=-200+2.5: 5 :200+2.5;
% x=1:81;
% y=1:81;
axis([-197.5 202.5 -197.5 202.5])
if isequal(plottype,'pcolor')
    if ndims (data_in_3_demention) ==3
a=pcolor(x,y,squeeze(data_in_3_demention(1,:,:)));%
    else
        a=pcolor(x,y,data_in_3_demention);%
    end
set(a, 'LineStyle','none');
%  clim([min(int) max(int)])
hold on
% b1=contourf(x,y,squeeze(data_in_3_demention(1,:,:))/1000,int,'LineStyle','none',FaceAlpha=1);%can show data
 colormap(gca,colormatinput);

hold on 
    if ndims(data_in_3_demention) ==3
[C,h]=contour(x,y,squeeze(data_in_3_demention(1,:,:)),textlable,'EdgeColor','w','ShowText','on',LineWidth=0.8);

       else
[C,h]=contour(x,y,data_in_3_demention,textlable,'EdgeColor','w','ShowText','on',LineWidth=0.8);
        end
clabel(C,h,'fontsize',9,'color','k','LabelSpacing',1200)

% xlim([-200 200])
%画圈圈
r = 100;%半径 
r1 = 200;%半径 
r3=65;
para1 = [0-r, 0-r, 2*r, 2*r];
para2 = [0-r1, 0-r1, 2*r1, 2*r1];
% para3 = [0-r3, 0-r3, 2*r3, 2*r3];
rectangle('Position', para1, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
rectangle('Position', para2, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
% rectangle('Position', para3, 'Curvature', [1 1],'EdgeColor','w', 'LineWidth',1.2);
axis equal
grid ("on")
colorbar
title(titlename)
elseif isequal(plottype,'contourf') 

    if ndims(data_in_3_demention) ==3
b1=contourf(x,y,squeeze(data_in_3_demention(1,:,:)),int,'LineStyle','none',FaceAlpha=1);%can show data
hold on 
[C,h]=contour(x,y,squeeze(data_in_3_demention(1,:,:)),textlable,'EdgeColor','w','ShowText','on',LineWidth=0.8);

       else
    b1=contourf(x,y,data_in_3_demention,int,'LineStyle','none',FaceAlpha=1);%can show data
    hold on 
[ C,h]=contour(x,y,data_in_3_demention,textlable,'EdgeColor','w','ShowText','on',LineWidth=0.8);

        end
    colormap(gca,colormatinput);


clabel(C,h,'fontsize',9,'color','w','LabelSpacing',1200)
% hold on
% scatter(station(:,1)/1000,station(:,2)/1000,5,'w',"filled");

%画圈圈
r = 100;%半径 
r1 = 200;%半径 
% r3=60;
para1 = [0-r, 0-r, 2*r, 2*r];
para2 = [0-r1, 0-r1, 2*r1, 2*r1];
% para3 = [0-r3, 0-r3, 2*r3, 2*r3];
rectangle('Position', para1, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
rectangle('Position', para2, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
% rectangle('Position', para3, 'Curvature', [1 1],'EdgeColor','w', 'LineWidth',1.2);
axis equal
grid ("on")
colorbar
title(titlename)
end
set(gca,"FontSize",18);











elseif nargin==5
x=-200+2.5: 5 :200+2.5;
y=-200+2.5: 5 :200+2.5;
axis([-197.5 202.5 -197.5 202.5])

if isequal(plottype,'pcolor')
    if ndims (data_in_3_demention) ==3
a=pcolor(x,y,squeeze(data_in_3_demention(1,:,:)));%
    else
        a=pcolor(x,y,data_in_3_demention);%
    end
set(a, 'LineStyle','none');
%  clim([min(int) max(int)])
% hold on
% b1=contourf(x,y,squeeze(data_in_3_demention(1,:,:))/1000,int,'LineStyle','none',FaceAlpha=1);%can show data
 colormap(gca,colormatinput);

% hold on 
% [C,h]=contour(x,y,squeeze(data_in_3_demention(1,:,:))/1000,textlable,'EdgeColor','k','ShowText','on',LineWidth=0.8);
% 
% clabel(C,h,'fontsize',10,'color','k','LabelSpacing',1200)
%画圈圈
r = 100;%半径 
r1 = 200;%半径 
r3=60;
para1 = [0-r, 0-r, 2*r, 2*r];
para2 = [0-r1, 0-r1, 2*r1, 2*r1];
% para3 = [0-r3, 0-r3, 2*r3, 2*r3];
rectangle('Position', para1, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
rectangle('Position', para2, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
% rectangle('Position', para3, 'Curvature', [1 1],'EdgeColor','w', 'LineWidth',1.2);
% axis equal
grid ("on")
% colorbar
title(titlename)







elseif isequal(plottype,'contourf') 
         if ndims(data_in_3_demention) ==3
b1=contourf(x,y,squeeze(data_in_3_demention(1,:,:)),int,'LineStyle','none',FaceAlpha=1);%can show data
       else
    b1=contourf(x,y,data_in_3_demention,int,'LineStyle','none',FaceAlpha=1);%can show data
        end
    colormap(gca,colormatinput);

% hold on 
% [C,h]=contour(x,y,squeeze(data_in_3_demention(1,:,:))/1000,textlable,'EdgeColor','k','ShowText','on',LineWidth=0.8);
% 
% clabel(C,h,'fontsize',10,'color','k','LabelSpacing',1200)
% hold on
% scatter(station(:,1)/1000,station(:,2)/1000,5,'w',"filled");

%画圈圈
r = 100;%半径 
r1 = 200;%半径 
r3=60;
para1 = [0-r, 0-r, 2*r, 2*r];
para2 = [0-r1, 0-r1, 2*r1, 2*r1];
% para3 = [0-r3, 0-r3, 2*r3, 2*r3];
rectangle('Position', para1, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
rectangle('Position', para2, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
% rectangle('Position', para3, 'Curvature', [1 1],'EdgeColor','w', 'LineWidth',1.2);
axis equal
grid ("on")
colorbar
title(titlename)
end
set(gca,"FontSize",18);
end
end