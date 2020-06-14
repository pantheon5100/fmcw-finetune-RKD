function [begin, finish, num]=framesplit(f)
    range_bin = size(f, 1);
    doppler_bin = size(f, 2);
    total_frame = size(f, 3);
    begin = 0;
%     frame = 0;
    num = 0;
    temp = 0;
    finish = 0;
    flag = 0;
    flag_1 = 0;
    sum_val = zeros(1, total_frame);
    for i = 1 : total_frame
        sum_val(1, i) = sum(sum(f(:, :, i)));
    end
    [max_val, pos] = max(sum_val);
    for i = 1 : total_frame
        if ~isequal(f(:,:,i), zeros(range_bin, doppler_bin)) && ~flag
            begin = i;
            flag = 1;
        end
        if flag && ~isequal(f(:,:,i), zeros(range_bin, doppler_bin))
            finish = i;
            if (finish - temp)> 1 && temp ~= 0 
                flag = 0;
            end
            temp = finish;
            if sum_val(1,finish) == max_val
                flag_1 = 1;
            end
        end
        if flag_1 && isequal(f(:,:,i), zeros(range_bin, doppler_bin))
            break
        end
    end
    if abs(begin-pos)>6
        begin = pos - 7;
    end
    if begin<=0
        num = 1;
    end
    if abs(finish-pos)>6
        finish = pos + 7;
    end

    frame = floor((finish+begin)/2);
    begin = frame - 6;

    finish = frame + 6;
    if begin < 1
        begin = 1;
        finish = begin + 12;
    end
    
    if finish>32
        finish = 32;
        begin = finish - 12;
    end
        
end