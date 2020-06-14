%%���ݸ�ʽΪ���Ƽ�����������eg ���� (2)1.mat�����������Ϊfa��fb����С��10*32*16
clc;
clear all;
%%ԭʼ���������������ļ�·��
n1 = zeros(22, 1);
n2 = zeros(22, 1);
n3 = zeros(22, 1);
n4 = zeros(22, 1);
sum_1 = zeros(22, 13);
sum_2 = zeros(22, 13);
sum_3 = zeros(22, 13);
sum_4 = zeros(22, 13);
win_sz = 224;
han_win = hanning(win_sz);
nfft = win_sz;
nooverloop = win_sz - 6;
fs = 128 / 32 * 1000;
for k = 22 : 22
    
    str_len = 2;
    if (k==3) || (k==4) || (k==9) ||(k==10) || (k==15) || (k == 16) || (k==21)||(k==22)
        str_len = 3;
    end

    path(17,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\dl\ǰ��\'};
    path(18,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\dl\��\'};
    path(19,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\dl\����\'};
    path(20,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\dl\����\'};
    path(21,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\dl\��ָ��\'};
    path(22,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\dl\��ָ��\'};
    
    namelist=dir([char(path(k)),'*.bin']);
    addr =char(path(k));
    i_=length(namelist);
    chirp_per_frame = 128;
    samples = 64;
    wrong_1 = 0;
    for i = 1 : i_
        %%
        %�õ�ʱ��������,reshape֮��1024*2048
        disp(i);
        fileName = [addr,namelist(i).name];
        retVal = readTSW16ant(fileName, samples, 4);
        [r1,r2, r3, r4] = f_4ant(retVal, samples);
        %%
        %2dfft���㣬֡��16������1��fa��ʾ������2��fb��ʾ.
        n = 1; %%������ÿһ��ͼƬ16֡

        f1 = zeros(13,33,32);
        f2 = zeros(13,33,32);
        f3 = zeros(13,33,32);
        f4 = zeros(13,33,32);

        for j = 1 : 128 : 4096
            f1(:,:,n) = compare_data(r1(:,j:j+127));
            f2(:,:,n) = compare_data(r2(:,j:j+127));
            f3(:,:,n) = compare_data(r3(:,j:j+127));
            f4(:,:,n) = compare_data(r4(:,j:j+127));
            n = n + 1;
        end
        [begin1, finish, num] = framesplit(f1);
        fa = f1(:, :, begin1:finish);
        n1(k, 1) = n1(k, 1) + num;
        [begin2, finish, num] = framesplit(f2);
        fb = f2(:, :, begin2:finish);  
        n2(k, 1) = n2(k, 1) + num;
        [begin3, finish, num] = framesplit(f3);
        fc = f3(:, :, begin3:finish);
        n3(k, 1) = n3(k, 1) + num;
        [begin4, finish, num] = framesplit(f4);
        n4(k, 1) = n4(k, 1) + num;
        fd = f4(:, :, begin4:finish); 
        P1 = time_freq(r1, fa, begin1, han_win, fs, nfft, nooverloop);
        P1 = int8(P1);
        P2 = time_freq(r2, fb, begin2, han_win, fs, nfft, nooverloop);
        P2 = int8(P2);
        P3 = time_freq(r3, fc, begin3, han_win, fs, nfft, nooverloop);
        P3 = int8(P3);
        P4 = time_freq(r4, fd, begin4, han_win, fs, nfft, nooverloop);
        P4 = int8(P4);

     %%
        if k < 5
            str = 'zk';
        elseif (k>=5) && (k<11)
            str = 'lt';
        elseif (k>=11)&&(k<17)
            str = 'jc';
        elseif (k>=17)
            str = 'dl';
        end
        data_dir = ['C:\ZGY\dataset\time_freq_mean_split_134ant\RDM\',str];%% compare_data�ŵ��ǲ���130�ģ������relu�����������븺����compare_data_1��130
        %compare_data_11�Ѳ�����-130�ĵ�С��10��֡ȫ����Ϊ130.compare_data_12 ��11�Ļ����Ϲ�һ��
        data_name = namelist(i).name;
        data_name = data_name(1:length(data_name)-4);%%ȥ��.bin
        num_match = regexp(data_name, '\d+', 'split');
    %     temp = regexp(data_name, '\s+', 'split');  %û�пո�����
    %     sub_dir = char(temp(1)); %%�õ����ļ������֣��ļ��ṹΪC:\result\data\sub_dir;egC:\result\data\�����״�
        temp = data_name(1 : str_len);

        sub_dir = char(temp);
        data_save_path = [data_dir,'\',sub_dir];
        if exist (data_save_path,'dir') ~= 7
            mkdir(data_save_path);
        end

        data_save_name = [data_save_path,'\',data_name,'1','.mat'];
        save(data_save_name,'P1');
%         imwrite(uint8(P1), [data_save_path,'\',data_name,'1','.png']);

        data_save_name = [data_save_path,'\',data_name,'2','.mat'];
        save(data_save_name,'P2');
%         imwrite(uint8(P2), [data_save_path,'\',data_name,'2','.png']);

        data_save_name = [data_save_path,'\',data_name,'3','.mat'];
        save(data_save_name,'P3');
%         imwrite(uint8(P3), [data_save_path,'\',data_name,'3','.png']);

        data_save_name = [data_save_path,'\',data_name,'4','.mat'];
        save(data_save_name,'P4');
%         imwrite(uint8(P4), [data_save_path,'\',data_name,'4','.png']);
    end
end       