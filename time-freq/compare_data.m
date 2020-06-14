function c_d = compare_data(data)
    d1 = fft_win_remove_zero_doppler(data);
    d1 = d1 .* d1;
    aptu_d1 = 10*log10(abs(fftshift(d1,2)));
    d2 = fft_win(data);
    d2 = d2 .* d2;
    aptu_d2 = 10*log10(abs(fftshift(d2,2)));
    c_d = aptu_d1(1:13, 64-16:64+16);
    num = 13 * 33;

end