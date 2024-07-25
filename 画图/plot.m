%% test

%选取的站点

% 读取Excel文件
filename = 'D:\zhanghuiyi-2021-2024\02_python\demo_montecarlo.csv';
data = readtable(filename);
sta=[];
sta(:,1) = data.lon;
sta(:,2) = data.lat;


 
% 原文件用来查找名称
filename = 'D:\zhanghuiyi-2021-2024\01_sta_network_evaluation\1-1站点初步筛选\test_stadata0626.xlsx';
data2 = readtable(filename);

% 提取站点名称和经纬度
site_names = data2{:, 1};
latitudes = str2double(data2{:, 3});
longitudes = str2double(data2{:, 2});



% 初始化存储站点名称的数组
sta_names = cell(size(sta, 1), 1);

% 查找并匹配名称
for i = 1:size(sta, 1)
    lat = sta(i, 2);
    lon = sta(i, 1);
    
    % 查找匹配的站点名称
    idx = find(abs(latitudes - lat) < 1e-3 & abs(longitudes - lon) < 1e-3);
    if ~isempty(idx)
        sta_names{i} = site_names{idx};
    else
        sta_names{i} = 'Unknown';
    end
end



%% 模拟结果

load D:\zhanghuiyi-2021-2024\01_sta_network_evaluation\1-4绘图\test_20240626.mat
meanlon=mean(sta(:,1));
meanlat=mean(sta(:,2));
CustomColormap='jet';
figure;%水平误差
montecalor_show(rmses_r(1,:,:)/1000,0:0.05:3,CustomColormap, '水平误差（z=7 km）','contourf',[0 0.05  0.3 0.6 1 ])
% montecalor_show(rmses_r(3,:,:)/1000,0:0.05:3,CustomColormap, '高度误差（z=7 km）','contourf',[0 0.05  0.3 0.6 1 ])

%转为km
d=lla2flat([sta(:,2),sta(:,1),sta(:,1)*0],[meanlat meanlon],0,0);
hold on 
scatter(d(:,2)/1000,d(:,1)/1000,50,'w','^','filled',MarkerEdgeColor='k');% 193 MHz
xlabel('E-W(km)');
ylabel('S-N(km)');
xlim([-150 150])
ylim([-150 150])

figure;%高度误差
% montecalor_show(rmses_r(1,:,:)/1000,0:0.05:3,CustomColormap, '水平误差（z=7 km）','contourf',[0 0.05  0.3 0.6 1 ])
montecalor_show(rmses_r(3,:,:)/1000,0:0.05:3,CustomColormap, '高度误差（z=7 km）','contourf',[0 0.05  0.3 0.6 1 ])

%转为km
d=lla2flat([sta(:,2),sta(:,1),sta(:,1)*0],[meanlat meanlon],0,0);
hold on 
scatter(d(:,2)/1000,d(:,1)/1000,50,'w','^','filled',MarkerEdgeColor='k');% 193 MHz
xlabel('E-W(km)');
ylabel('S-N(km)');
xlim([-150 150])
ylim([-150 150])

% %% 叠加shp
% gx1=shaperead('D:\important_copy\站网评估-检查\计算遮挡角\重庆\重庆市\重庆市.shp','UseGeoCoords',true);  
% figure
% set(gcf,"Position",[10 10 700 700])
% geoshow(gx1,'FaceColor',[1,1.0,1],'EdgeColor','black','LineWidth',1.0);%
% set(gcf,"Position",[10 10 700 700])
% 
% xlabel('longitude(°E)');
% ylabel('latitude(°N)');
% set(gca,"FontSize",12,'FontName','Times New Roman')
% hold on
% scatter(sta(:,1),sta(:,2),50,'r','^','filled',MarkerEdgeColor='k');% 
% box on
% grid on
% axis equal
% 
% 
% 
% hold on
% 
% for i=1:length(name)
% text(sta(i,1)-0.03,sta(i,2)+0.04,string(name(i)),"FontSize",10)%写站名
% end
% ylim([28 30])
% xlim([106 108])
% 
% %116.7308186	26.28293222
% %画圈圈
% r = 1;%半径 
% r1 = 1.5;%半径 
% r3=65;
% hold on
% meanlon=mean(sta(:,1));
% meanlat=mean(sta(:,2));
% % scatter(meanlon,meanlat,50,'k','^',MarkerEdgeColor='k');% 193 MHz
% % text(meanlon+0.02,meanlat+0.02,'sta middle',"FontSize",10)%写站名
% para1 = [meanlon-r,  meanlat-r, 2*r, 2*r];
% para2 = [meanlon-r1, meanlat-r1, 2*r1, 2*r1];
% % para3 = [0-r3, 0-r3, 2*r3, 2*r3];
% rectangle('Position', para1, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
% rectangle('Position', para2, 'Curvature', [1 1],'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1);
% annotation('textbox',...
%     [0.651428571428571 0.241857143580913 0.0885714263575418 0.0385714278476579],...
%     'String',{'100km'},...
%     'LineStyle','none');
% annotation('textbox',...
%     [0.742857142857143 0.159000000723771 0.0885714263575418 0.0385714278476579],...
%     'String',{'150km'},...
%     'LineStyle','none');

