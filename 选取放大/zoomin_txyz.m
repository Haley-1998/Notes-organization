function zoomin_txyz(init_data, init_cgdata, corarea)

% 原始数据init_data格式为txzy, init_cgdata为地闪数据，可为空，corarea为所需放大的投影子图，原始图像为时间-高度和三个方向的投影。

% 如果corarea为空，则默认选择'T-Z'
if isempty(corarea)
    corarea='T-Z';
end
% 获取当前坐标轴的x和y刻度值
x=get(gca,'xtick'); 
y=get(gca,'ytick');

% 计算x和y坐标的最大值和最小值
xmax=max(x);xmin=min(x);
ymax=max(y);ymin=min(y);

% 根据选择的坐标区间进行数据筛选
switch corarea
    case 'T-Z'
        t = init_data(:, 1); % 获取时间列
        ind = find(xmin <= t & t <= xmax); % 找到时间在xmin和xmax之间的索引
        if isempty(ind)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data11 = init_data(ind, :); % 根据索引筛选数据

    ind2 = find(ymin <= data11(:, 4) & data11(:, 4) <= ymax); % 找到高度在ymin和ymax之间的索引
        if isempty(ind2)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data2 = data11(ind2, :); % 根据索引再次筛选数据

    case 'X-Y'
        ind = find(xmin <= init_data(:, 2) & init_data(:, 2) <= xmax); % 找到X坐标在xmin和xmax之间的索引
        if isempty(ind)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data11 = init_data(ind, :); % 根据索引筛选数据
        ind2 = find(ymin <= data11(:, 3) & data11(:, 3) <= ymax); % 找到Y坐标在ymin和ymax之间的索引
        if isempty(ind2)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data2 = data11(ind2, :); % 根据索引再次筛选数据

    case 'X-Z'
        ind = find(xmin <= init_data(:, 2) & init_data(:, 2) <= xmax); % 找到X坐标在xmin和xmax之间的索引
        if isempty(ind)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data11 = init_data(ind, :); % 根据索引筛选数据
        ind2 = find(ymin <= data11(:, 4) & data11(:, 4) <= ymax); % 找到Z坐标在ymin和ymax之间的索引
        if isempty(ind2)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data2 = data11(ind2, :); % 根据索引再次筛选数据

    case 'Z-Y'
        ind = find(xmin <= init_data(:, 4) & init_data(:, 4) <= xmax); % 找到Z坐标在xmin和xmax之间的索引
        if isempty(ind)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data11 = init_data(ind, :); % 根据索引筛选数据
        ind2 = find(ymin <= data11(:, 3) & data11(:, 3) <= ymax); % 找到Y坐标在ymin和ymax之间的索引
        if isempty(ind2)
            msgbox('坐标区选取错误或无数据', '信息框', 'warn');
            data2 = 0;
            return
        end
        data2 = data11(ind2, :); % 根据索引再次筛选数据
end

% 绘制放大后的图形
figure
plot_lma_ghmlls_wave(data2,init_cgdata,[],[],[],[],[],[],[]);




% 内嵌的绘图函数
    function plot_lma_ghmlls_wave(cor_flash,cor_LLS,datetime,WAVE_t,WAVE_plot,area,h,curt,cur)
        time=cor_flash(:,1);
        x   =cor_flash(:,2);
        y   =cor_flash(:,3);
        z   =cor_flash(:,4);
        lat0=23.56343219;
        lon0=113.47420229;
        alt0=43.78;
        if ~isempty(cor_LLS)
            cor_time=cor_LLS(:,8);
            cor_lon   =cor_LLS(:,10);
            cor_lat   =cor_LLS(:,9);
            cor_z   =cor_LLS(:,12);
            cor_curr   =cor_LLS(:,11);
            cur_name=num2str(round(cor_curr));

            ind_pos=find(cor_curr>0);
            ind_neg=find(cor_curr<0);
            ind_35=find( cor_curr>-35);
            ind_75=find(cor_curr<-75);
            ind_100=find(cor_curr<-100);
            ind_150=find(cor_curr<-150);
            distance = lla2flat( [cor_lat cor_lon cor_z], [lat0 lon0], 0, alt0 );
            cor_x=distance(:,2)/1000;
            cor_y=distance(:,1)/1000;

        else
            cor_time=[];
            cor_x   =[];
            cor_y   =[];
            cor_z   =[];
            cor_curr=[];
            cur_name=[];
            ind_pos=[];
            ind_neg=[];
            ind_35=[];
            ind_75=[];
            ind_100=[];
            ind_150=[];
        end

        low_z=0;
        up_z=20;

        low_x=min(x);
        up_x= max(x);
        low_y=min(y) ;
        up_y=max(y);

        maker_size=7;%----------------------改
        maker_size2=20;%----------------------改
        hm=second_change(min(time));
        hm2=second_change(max(time));
        hm3=second_change(max(time+28800));

        color=cor_flash(:,1);
        set(gcf, 'position', [0 0 600 800]);
        clf
        %%%------------time-z---------------------------------------
        axes1=axes('Position',[0.09 0.79 0.85 0.16],'Parent',gcf);
        if isempty(cor_LLS)
            scatter( time,z,maker_size,color,'filled' );
        else
            scatter( time-min(time),z,maker_size,color,'filled' );
        end
        hold on
        %         scatter( cor_time(ind_pos)-min(time),cor_z(ind_pos),abs(cor_curr(ind_pos)),'r','filled','^' );
        scatter( cor_time(ind_neg)-min(time),cor_z(ind_neg),abs(cor_curr(ind_neg)),'b','filled','^' );
        scatter( cor_time(ind_35)-min(time),cor_z(ind_35),abs(cor_curr(ind_35)),'r','filled','^','MarkerEdgeColor','k' );
        scatter( cor_time(ind_75)-min(time),cor_z(ind_75),abs(cor_curr(ind_75)),'g','filled','^','MarkerEdgeColor','k' );
        scatter( cor_time(ind_100)-min(time),cor_z(ind_100),abs(cor_curr(ind_100)),[0.69,0.15,0.81],'filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_time(ind_150)-min(time),cor_z(ind_150),abs(cor_curr(ind_150)),'black','filled','^' );
        set(gca,'fontsize',11)
        hold on
        for n=1:length(ind_neg)
            text(cor_time(ind_neg(n))-min(time)-0.012,cor_z(ind_neg(n))+1.7,cur_name(ind_neg(n),:),"FontSize",10)
        end

        colormap jet
        set(gcf,'color','w');
        set(gca,'box','on');
        set(gca,'YLim',[low_z up_z]);
        axis_position =axis;
        set(get(axes1,'ylabel'),'string','Altitude (km)','fontsize',11);
        set(get(axes1,'xlabel'),'string','Time (s)','fontsize',11);
        if ~isequal(hm(1:6),hm2(1:6))
            title([datetime,'  ',hm(1:2),':',hm(3:4),':',hm(5:6),'-',hm2(1:2),':',hm2(3:4),':',hm2(5:6),' UTC']);
        else
            title([datetime,'  ',hm(1:2),':',hm(3:4),':',hm(5:6),' UTC'])
        end
        if ~isempty(cur)
            subtitle(['area-',num2str(area),' init h=',num2str(h),' cur t=',num2str(curt),' cur=',num2str(cur)])
        end


        if ~isempty(WAVE_t)
            yyaxis right

            plot(WAVE_t-28837-min(time),WAVE_plot);

        end


        %%%-------------x-y--------------------------------------
        axes2=axes('Position',[0.09 0.05 0.55 0.48],'Parent',gcf);
        scatter(x,y,maker_size,color,'fill');
        hold on
        %         scatter( cor_x(ind_pos),cor_y(ind_pos),abs(cor_curr(ind_pos)),'r','filled','^' );
        scatter( cor_x(ind_neg),cor_y(ind_neg),abs(cor_curr(ind_neg)),'b','filled','^' );
        scatter( cor_x(ind_35),cor_y(ind_35),abs(cor_curr(ind_35)),'r','filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_x(ind_75),cor_y(ind_75),abs(cor_curr(ind_75)),'g','filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_x(ind_100),cor_y(ind_100),abs(cor_curr(ind_100)),[0.69,0.15,0.81],'filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_x(ind_150),cor_y(ind_150),abs(cor_curr(ind_150)),'black','filled','^' );
        hold on;
        LMA_LOC=[23.5843449000000	113.501350400000	67.0100000000000
            23.5010757000000	113.294357300000	48
            23.6759930000000	113.410835300000	42.8500000000000
            23.5681629000000	113.615188600000	43.7800000000000
            23.4826775000000	113.524070700000	53.1000000000000
            23.6213875000000	113.596183800000	59.7800000000000
            23.4213295000000	113.236045800000	31.7100000000000
            23.4398499000000	113.320770300000	43.0700000000000
            23.6076927000000	113.609603900000	47.3700000000000
            23.5941236000000	113.530462000000	67.0500000000000
            23.6610851000000	113.470962500000	71.8900000000000];
        distance2 = lla2flat( [LMA_LOC(:,1) LMA_LOC(:,2) LMA_LOC(:,3)], [lat0 lon0], 0, alt0 );
        loc_x=distance2(:,2)/1000;
        loc_y=distance2(:,1)/1000;
        scatter(loc_x, loc_y,10,"K",'filled','^');
        hold off;
        set(gca,'box','on');
        axis([low_x,up_x,low_y,up_y]);
        axis_position =axis;
        set(gca,'FontSize',11)  %是设置刻度字体大小
        set(get(axes2,'xlabel'),'string','west-east','fontsize',11);
        set(get(axes2,'ylabel'),'string','north-south','fontsize',11);
        set(gca,'fontsize',11)
        %%%-------x-z-------------------------------------------------------------
        axes3=axes('Position',[0.09 0.56 0.55 0.18],'Parent',gcf);
        scatter(x,z,maker_size,color,'fill');
        hold on

        %         scatter( cor_x(ind_pos),cor_z(ind_pos),abs(cor_curr(ind_pos)),'r','filled','^' );
        scatter( cor_x(ind_neg),cor_z(ind_neg),abs(cor_curr(ind_neg)),'b','filled','^' );
        scatter( cor_x(ind_35),cor_z(ind_35),abs(cor_curr(ind_35)),'r','filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_x(ind_75),cor_z(ind_75),abs(cor_curr(ind_75)),'g','filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_x(ind_100),cor_z(ind_100),abs(cor_curr(ind_100)),[0.69,0.15,0.81],'filled','^','MarkerEdgeColor','k' );
        scatter( cor_x(ind_150),cor_z(ind_150),abs(cor_curr(ind_150)),'black','filled','^' );
        %         axis([low_x,up_x,low_z,up_z]);
        set(gca,'XLim',[low_x up_x]);
        set(gca,'YLim',[low_z up_z]);
        set(gca,'box','on');
        %set(gca, 'XTick',[] );
        %set(gca, 'YTick',0:5:20  );
        axis_position =axis;
        set(get(axes3,'ylabel'),'string','Altitude(km)','fontsize',11);
        set(gca,'fontsize',11)
        %%%----------------z-y--------------------------------------------------------
        axes4=axes('Position',[0.72 0.05 0.22 0.48],'Parent',gcf);
        scatter(z,y,maker_size,color,'fill');
        hold on
        scatter( cor_z(ind_neg),cor_y(ind_neg),abs(cor_curr(ind_neg)),'b','filled','^' );
        scatter( cor_z(ind_35),cor_y(ind_35),abs(cor_curr(ind_35)),'r','filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_z(ind_75),cor_y(ind_75),abs(cor_curr(ind_75)),'g','filled','^' ,'MarkerEdgeColor','k');
        scatter( cor_z(ind_100),cor_y(ind_100),abs(cor_curr(ind_100)),[0.69,0.15,0.81],'filled','^','MarkerEdgeColor','k' );
        scatter( cor_z(ind_150),cor_y(ind_150),abs(cor_curr(ind_150)),'black','filled','^' );
        set(gca,'YLim',[low_y up_y]);
        set(gca,'XLim',[low_z up_z]);
        set(gca,'box','on');
        axis_position =axis;
        set(get(axes4,'xlabel'),'string','Altitude(km)','fontsize',11);
        set(gca,'fontsize',11)
        %%%------------------------------------------------------------------------
        axes5=axes('Position',[0.72 0.56 0.22 0.18],'Parent',gcf);
        h=histogram(z,0:0.5:round(max(z)),'FaceColor',[0.5 0.5 0.5],...
            'EdgeColor','none','FaceAlpha',0.7,'Orientation','horizontal');
        set(gca,'YLim',[low_z up_z]);
        set(gca,'box','on')
        axis_position =axis;
        text(axis_position(2)-0.085*(axis_position(2)-axis_position(1)),axis_position(3)+0.97*(axis_position(4)-axis_position(3)),[num2str(size(time,1)) ' pts'], ...
            'verticalalignment', 'top', 'horizontalalignment', 'right', 'fontsize', 9);
        set(gca,'fontsize',11)



        function Output = second_change(num)

            hour = floor(num/3600);              % floor: 向下取整
            minute = floor(mod(num,3600)/60);  % mod： 求余数
            second = num - 3600*hour - 60*minute;

            if hour < 10
                hour = ['0',mat2str(hour)];      % mat2str：将double转化为字符串
            else
                hour = mat2str(hour);
            end

            if minute < 10
                minute = ['0',mat2str(minute)];
            else
                minute = mat2str(minute);
            end

            if second < 10
                second = ['0',mat2str(second)];
            else
                second = mat2str(second);
            end

            Output = [hour,minute,second];
    end
end

end
