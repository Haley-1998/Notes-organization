## 绘制GIF和AVI核心代码
 ```matlab
%===============================gif=====================================================
 
figure;
for i = 1:size(fn, 1)
 
    % plot
    %
    %
    currFrame = getframe(gcf);
    I = frame2im(currFrame);
    [I, map] = rgb2ind(I, 256);
 
    gif_filename = fullfile(save_path, 'output_animation.gif');
    if i == 1
        imwrite(I, map, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(I, map, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
 
end
 
%=======================================================================================
%
%
%
%===============================avi=====================================================
 
vidObj = VideoWriter(fullfile(save_path, 'output_video'), 'Uncompressed AVI');
vidObj.FrameRate = 10;
open(vidObj);
 
figure;
for i = 1:size(fn, 1)
    % plot
    %
    %
    currFrame = getframe(gcf);
    writeVideo(vidObj, currFrame);
 
end
close(vidObj);
%=======================================================================================
 ```
## 1、绘制GIF
 ```matlab
function gif_plot(data_path, fn, save_path)
    % GIF_PLOT Generates a GIF from data.————绘制多个文件数据
    %
    % data_path: Path to the data files
    % fn: List of filenames to process
    % save_path: Path to save the generated GIF
 
    % Initialize a figure for plotting
    figure;
    pic_num = 1;
 
    % Loop through each filename in the list 'fn'
    for i = 1:size(fn, 1)
        % Construct the full file path—
        fnn = fn(i, :);
        load(fullfile(data_path, fnn));
        
        % Extract names for saving and titling
        save_name = fnn(end-17:end-10); % ———————自定义 filename for saving
        title_name = fnn(end-17:end-4); % Extracts the specific part of the filename for the title
 
 
        
        % Plot the data
        % ================================自定义绘图方式===============================================
        % ============================================================================================
        contour(cr_lon, cr_lat, cr_data, 0:5:65, 'LineColor', 'none', 'FaceAlpha', 1);
        
        % Set axis limits and labels
        axis([min(cr_lon) max(cr_lon) min(cr_lat) max(cr_lat)]);
        title(title_name);
        axis equal;
        xlabel('Longitude');
        ylabel('Latitude');
        box on;
        colormap(parula); % Default colormap, can be changed as needed
        colorbar;
        clim([0 65]);
        % ============================================================================================
        % ============================================================================================
 
        % Capture the current frame
        currFrame = getframe(gcf);
        I = frame2im(currFrame);
        [I, map] = rgb2ind(I, 256);
        
        % Write to GIF————————0.1为0.1秒一帧
        if pic_num == 1
            imwrite(I, map, fullfile(save_path, [save_name, '.gif']), 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
        else
            imwrite(I, map, fullfile(save_path, [save_name, '.gif']), 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
        
        % Increment the picture number for the next iteration
        pic_num = pic_num + 1;
    end
end
```
## 2、 绘制AVI
 ```matlab
function generate_radar_video(data_path, fn, save_path)
    % GENERATE_RADAR_VIDEO Generates a video from radar data.
    %
    % data_path: Path to the radar data files
    % fn: List of filenames to process
    % save_path: Path to save the generated video
 
    % Create a video writer object
    vidObj = VideoWriter(fullfile(save_path, 'radar_video'), 'Uncompressed AVI');
    vidObj.FrameRate = 10; % Play speed (how many pictures in one second)
    open(vidObj);
 
    figure;
    set(gcf, 'Position', [10, 10, 1200, 1200]);
 
    % Loop through each file
    for i = 1:size(fn, 1)
 
        %==================自定义绘图====================================================
        %================================================================================
        try
            clf;
            file = fullfile(data_path, fn(i, :));
            g = py.cinrad.io.StandardData(file);
            rl = py.list(g.iter_tilt(230, 'REF'));
            cr = py.cinrad.calc.quick_cr(rl);
            CR = py.dict(cr);
            Cr = double(CR{'CR'}.data);
            Lon = double(CR{'CR'}.coords.variables{'longitude'}.data);
            Lat = double(CR{'CR'}.coords.variables{'latitude'}.data);
            Cr(Cr < 0) = nan;
            [LON, LAT] = meshgrid(Lon, Lat); % Create a grid
 
            % Plot the radar data
            pcolor(LON, LAT, Cr);
            colormap(parula); % Default colormap, can be changed as needed
            shading flat;
            grid on;
            hold on;
 
            % Additional data points and plots (modify as needed)
            % scatter(...); % Example: scatter plot for additional data points
 
            % Plot settings
            axis equal;
            clim([0, 75]);
            colorbar('EastOutside');
            h = colorbar;
            set(get(h, 'title'), 'string', '(dBZ)'); % Set colorbar unit
            set(h, 'YTick', 0:10:70); % Set colorbar ticks
            set(h, 'YTickLabel', arrayfun(@num2str, 0:10:70, 'UniformOutput', false));
            xlabel('Longitude');
            ylabel('Latitude');
            box on;
            xlim([min(Lon), max(Lon)]);
            ylim([min(Lat), max(Lat)]);
            title(['CR ', fn(i, 16:23), '-', fn(i, 24:29)]);
            subtitle(['File: ', fn(i, :)]);
 
            hold off;
        %================================================================================
        %================================================================================
 
 
            % Capture the current frame
            currFrame = getframe(gcf);
 
            % Write the frame to the video
            writeVideo(vidObj, currFrame);
 
        catch
            disp(['Error processing file: ', fn(i, :)]);
            continue;
        end
    end
 
    % Close the video writer object
    close(vidObj);
end

```
