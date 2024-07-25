## 新建文件夹mkdir
```matlab
% 定义保存路径
savepath = 'D:\2024\';
 
% 如果保存路径不存在，则创建该路径
if ~exist(savepath, 'dir')
    mkdir(savepath)
end
```

## 解压文件gunzip，删除文件delete
```matlab
% 指定的目录路径
sl_dir
 
% 获取目录下所有的文件夹
folders = dir(sl_dir);
 
% 从第3个文件夹开始（因为前两个是 . 和 ..）
for i = 3:size(folders, 1)
 
    % 获取文件夹的完整路径
    file_path = fullfile(folders(i).folder, folders(i).name);
    
    % 保存文件夹的名字
    save_name = folders(i).name;                               
    
    % 获取指定路径下的所有.gz文件
    namelist = ls([file_path, '\*.gz']);
    
    % 如果namelist非空，则进行解压操作
    if ~isempty(namelist)                                     
        % 解压所有的.gz文件到extrac_dir目录下
        gunzip(fullfile(file_path, string(namelist)), extrac_dir);
    end
    
    % 显示成功解压的信息
    disp(['success folder ', save_name]);
                   
end
 
% 获取解压目录下的所有.dat文件
    namelist2 = ls([extrac_dir, '\*.dat']);
 
% 删除extrac_dir目录下所有的.dat文件
    delete([extrac_dir, '\*.dat']);    
```

