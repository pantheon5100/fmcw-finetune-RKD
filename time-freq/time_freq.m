function [P] = time_freq(r1, f, begin, han_win, fs, nfft, nooverloop)
    f_r1 = fft(r1);
    sum_1 = zeros(13, 13);
    for c = 1 : 13
        sum_1(:, c) = sum(f(:, :, c), 2);
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
    if num == 0
        range_1 = 2;
    end
    [~, ~, ~, P1] = spectrogram(f_r1(range_1-1,(begin-1)*128+1:(begin+12)*128), han_win, nooverloop, nfft, fs);
    P1 = 10*log10((abs(fftshift(P1, 1))));

    P(:, :, 1) = P1(:,9:232);
    [~, ~, ~, P1] = spectrogram(f_r1(range_1,(begin-1)*128+1:(begin+12)*128), han_win, nooverloop, nfft, fs);
    P1 = 10*log10((abs(fftshift(P1, 1))));
    P(:, :, 2) = P1(:, 9:232);
    [~, ~, ~, P1] = spectrogram(f_r1(range_1+1,(begin-1)*128+1:(begin+12)*128), han_win, nooverloop, nfft, fs);
    P1 = 10*log10((abs(fftshift(P1, 1))));
    P(:, :, 3) = P1(:, 9:232);
%     max_val = max(max(max(P)));
%     P = P / max_val * 255;
    for i = 1 : size(P, 1)
        for j = 1 : size(P, 2)
            for k = 1 : size(P, 3)
                if(P(i, j, k)<40)
                    P(i, j, k) = 0;
                end
            end
        end
    end
    
    
end