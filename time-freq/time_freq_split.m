path = cell(13,1);
path(1,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zgy\按下\'};
path(2,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zgy\提起\'};
path(3,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zgy\五指握\'};
path(4,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zgy\五指张\'};
path(5,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zk\前拨\'};
path(6,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zk\后拨\'};
path(7,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zgy\前拨\'};
path(8,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zgy\后拨\'};
path(9,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zk\按下\'};
path(10,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zk\提起\'};
path(11,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zk\五指握\'};
path(12,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\4.6 data\zk\五指张\'};
path(13,1) = {'C:\ti\mmwave_dfp_01_00_00_01\rf_eval\radarstudio\PostProc\'};
k = 6;
namelist=dir([char(path(k)),'*.bin']);
addr =char(path(k));
i =5;
samples = 64;
win_sz = 128;
han_win = hanning(win_sz);
nfft = win_sz;
nooverloop = win_sz - 8;
fileName = [addr,namelist(i).name];
retVal = readTSW16(fileName, samples);
[r1, r2] = f(retVal, samples);
f_r1 = fft(r1);
f_r2 = fft(r2);
name = namelist(i).name(1:end-4);
fs = 128 / 32 * 1000;
f1 = zeros(13,33,32);
f2 = zeros(13,33,32);
n = 1;
for j = 1 : 128 : 4096
    f1(:,:,n) = compare_data(r1(:,j:j+127));

    f2(:,:,n) = compare_data(r2(:,j:j+127));

    n = n + 1;
end

[begin1, finish1, a] = framesplit(f1);
fa = f1(:, :, begin1:finish1);
[begin2, finish2, a] = framesplit(f2);
fb = f2(:, :, begin2:finish2);  
for c = 1 : 13
    sum_1(:, c) = sum(fa(:, :, c), 2);
    sum_2(:, c) = sum(fb(:, :, c), 2);
end
num_1 = zeros(13, 13);
num_2 = zeros(13, 13);
for c = 1 : 13
    for i = 1 : 13
        for j = 1 : 33
            if(fa(i, j, c) ~= 0)
                num_1(i, c) = num_1(i, c) + 1;
            end
            if fb(i, j, c) ~= 0
                num_2(i, c) = num_2(i, c) + 1;
            end
        end
    end
end

[~, pos] = max(sum_1);
num = 0;
fix_pos = zeros(1, 13);
for c = 1 : 13
    if(pos(1, c)~=1)
        num = num + 1;
        fix_pos(1, num) = pos(1, c);
    end
end
fix_pos = fix_pos(1, 1:num);

range_1 = round(mean(fix_pos));
if(range_1 < 1)
    range_1 = 2
end
figure(30)
sumf_1 = f_r1(range_1-1, (begin1-1)*128+1:(begin1+12)*128) + f_r1(range_1, (begin1-1)*128+1:(begin1+12)*128) + f_r1(range_1+1, (begin1-1)*128+1:(begin1+12)*128);
sumf_2 = f_r2(range_1-1, (begin1-1)*128+1:(begin1+12)*128) + f_r2(range_1, (begin1-1)*128+1:(begin1+12)*128) + f_r2(range_1+1, (begin1-1)*128+1:(begin1+12)*128);
sumf_1 = sumf_1 / 3;
sumf_2 = sumf_2 / 3;

[S1, F1, T1, P1] = spectrogram(sumf_1, han_win, nooverloop, nfft, fs);
[S2, F2, T2, P2] = spectrogram(sumf_2, han_win, nooverloop, nfft, fs);
subplot(211)

imagesc(T1, F1, 10*log10((abs(fftshift(P1, 1)))), [35, 60])
set(gca, 'YDir', 'normal')
xlabel('Time(secs)')
ylabel('Freq(Hz)')
title(['ant1 time-freq-sum fig', name])
subplot(212)

imagesc(T2, F2, 10*log10((abs(fftshift(P2, 1)))), [35, 60])
set(gca, 'YDir', 'normal')
xlabel('Time(secs)')
ylabel('Freq(Hz)')
title(['ant2 time-freq-sum fig', name])
figure(33)
[S1, F1, T1, P1] = spectrogram(f_r1(range_1-1,(begin1-1)*128+1:(begin1+12)*128), han_win, nooverloop, nfft, fs);
subplot(311)
imagesc(T1, F1, 10*log10((abs(fftshift(P1, 1)))), [35, 60])
set(gca, 'YDir', 'normal')
xlabel('Time(secs)')
ylabel('Freq(Hz)')
title(['ant1 time-freq fig', name])
[S1, F1, T1, P1] = spectrogram(f_r1(range_1,(begin1-1)*128+1:(begin1+12)*128), han_win, nooverloop, nfft, fs);
subplot(312)
P1 = 10*log10((abs(fftshift(P1, 1))));
imagesc(T1, F1, P1, [35, 60])
set(gca, 'YDir', 'normal')
xlabel('Time(secs)')
ylabel('Freq(Hz)')
title(['ant1 time-freq fig', name])
[S1, F1, T1, P1] = spectrogram(f_r1(range_1+1,(begin1-1)*128+1:(begin1+12)*128), han_win, nooverloop, nfft, fs);
subplot(313)
imagesc(10*log10((abs(fftshift(P1, 1)))), [35, 60])
set(gca, 'YDir', 'normal')
xlabel('Time(secs)')
ylabel('Freq(Hz)')
title(['ant1 time-freq fig', name])
chirp = 128;

P = time_freq(r1, fa, begin1, han_win, fs, nfft, nooverloop);
figure(2)
subplot(211)
imagesc(P)
subplot(212)
imagesc(P(:, :, 1))

t = int8(P);