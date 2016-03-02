%%
function Img = RowToImg( RowVec )
% This function transfer a 1x3072 row vector into a 32x32x3 image.
    red = RowVec(1:1024);
    green = RowVec(1024+1:1024+1024);
    blue = RowVec(1024*2+1:1024*2+1024);
    Img = zeros(32,32,3);
    Img(:,:,1) = reshape(red',32,32)';
    Img(:,:,2) = reshape(green',32,32)';
    Img(:,:,3) = reshape(blue',32,32)';
end

