
%SelectUniformGroups.m
% 读取Excel文件
filename = 'test_stadata0626.xlsx';
data = readtable(filename);

% 提取站点名称和经纬度
site_names = data{:, 1};
latitudes = str2double(data{:, 3});
longitudes = str2double(data{:, 2});

% 设置中心经纬度
center_lat = 29;
center_lon = 107;

% 计算站点与中心点的距离
refEllipsoid = referenceEllipsoid('wgs84');
distances = distance(center_lat, center_lon, latitudes, longitudes, refEllipsoid) / 1000; % 转换为km

% 选取100km范围内的站点
within_range = distances <= 100;
selected_sites = site_names(within_range);
selected_latitudes = latitudes(within_range);
selected_longitudes = longitudes(within_range);

% 初始化结果存储
groups = {};
group_count = 0;
max_groups = 10;
num_sites = length(selected_sites);

% 设置组的站点数量
num_stations_in_group = 5; % 可变参数

% 计算所有站点之间的距离矩阵
distance_matrix = zeros(num_sites);
for i = 1:num_sites
    for j = i+1:num_sites
        distance_matrix(i,j) = distance(selected_latitudes(i), selected_longitudes(i), selected_latitudes(j), selected_longitudes(j), refEllipsoid) / 1000;
        distance_matrix(j,i) = distance_matrix(i,j);
    end
end

% 进行数据分组
for i = 1:num_sites-num_stations_in_group+1
    for j = i+1:num_sites-num_stations_in_group+2
        for k = j+1:num_sites-num_stations_in_group+3
            for m = k+1:num_sites-num_stations_in_group+4
                for n = m+1:num_sites-num_stations_in_group+5
                    group_indices = [i, j, k, m, n];
                    group_distances = distance_matrix(group_indices, group_indices);
                    
                    % 检查组内站点距离是否在20-70km范围内
                    valid_group = true;
                    for p = 1:num_stations_in_group
                        for q = p+1:num_stations_in_group
                            if group_distances(p,q) < 20 || group_distances(p,q) > 70
                                valid_group = false;
                                break;
                            end
                        end
                        if ~valid_group
                            break;
                        end
                    end

                    % 添加到组列表
                    if valid_group
                        group_count = group_count + 1;
                        groups{group_count} = struct('names', {selected_sites(group_indices)}, ...
                                                     'latitudes', selected_latitudes(group_indices), ...
                                                     'longitudes', selected_longitudes(group_indices), ...
                                                     'distances', group_distances);
                        if group_count >= max_groups
                            break;
                        end
                    end
                end
                if group_count >= max_groups
                    break;
                end
            end
            if group_count >= max_groups
                break;
            end
        end
        if group_count >= max_groups
            break;
        end
    end
    if group_count >= max_groups
        break;
    end
end

% 选择距离差异最小的10组
group_dist_diff = zeros(length(groups), 1);
for k = 1:length(groups)
    dist_diff = max(groups{k}.distances(:)) - min(groups{k}.distances(:));
    group_dist_diff(k) = dist_diff;
end

[~, idx] = sort(group_dist_diff);
final_groups = groups(idx(1:max_groups));

% 准备保存到Excel的数据
output_data = {};
for m = 1:length(final_groups)
    for n = 1:num_stations_in_group
        output_data = [output_data; {final_groups{m}.names{n}, final_groups{m}.latitudes(n), final_groups{m}.longitudes(n), 3, 11, 1, 15}];
    end
end

% 保存到Excel文件
output_filename = ['初选', datestr(now, 'yyyymmdd'), '.xlsx'];
output_table = cell2table(output_data, 'VariableNames', {'Name', '纬度', '经度', 'PitchAngle', 'RackHigh', 'AngleInterval', 'EfficacyScope'});
writetable(output_table, output_filename);

% 保存分组信息
for p = 1:max_groups
    eval(['group', num2str(p), ' = final_groups{p};']);
end

disp(['数据已保存到文件: ', output_filename]);

% 绘制每个组的站点在子图中
figure;

for m = 1:max_groups
    subplot(2, 5, m);
    scatter(groups{m}.longitudes, groups{m}.latitudes, 50, 'filled');
    title(['Group ', num2str(m)]);
    xlabel('经度');
    ylabel('纬度');
    grid on;
    axis equal
    % 在每个点的位置添加站点名称标签
    for n = 1:length(groups{m}.names)
        text(groups{m}.longitudes(n), groups{m}.latitudes(n), groups{m}.names{n}, 'FontSize', 8, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    end
end

% 调整整体布局
sgtitle('Station Groups Distribution');
