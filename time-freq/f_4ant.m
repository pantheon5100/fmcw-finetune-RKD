function [r1,r2, r3, r4]=f_4ant(x,samples)
    r1=x(1,:);
    r1=reshape(r1,samples,[]);
%     f1=log(abs(fftshift(fft2(r1))));
    r2=x(2,:);
    r2=reshape(r2,samples,[]);
    r3=x(3,:);
    r3=reshape(r3,samples,[]);
%     f1=log(abs(fftshift(fft2(r1))));
    r4=x(4,:);
    r4=reshape(r4,samples,[]);

