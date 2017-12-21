clear
filename='temp3'
I=imread([filename,'.jpg']);
R=double(I(:,:,1));
G=double(I(:,:,2));
B=double(I(:,:,3));
[Ru,Rz,Rv]=svd(R);
[Gu,Gz,Gv]=svd(G);
[Bu,Bz,Bv]=svd(B);
for k=1:2:25
R_new=Ru(:,1:k)*Rz(1:k,1:k)*Rv(:,1:k)';
G_new=Gu(:,1:k)*Gz(1:k,1:k)*Gv(:,1:k)';
B_new=Bu(:,1:k)*Bz(1:k,1:k)*Bv(:,1:k)';
I_new=cat(3,R_new,G_new,B_new);
imshow(uint8(I_new))
picname=[filename,'_',num2str(k) '.jpg'];%保存的文件名：如i=1时，picname=1.fig
    %hold on % 写后面的字时，不把前面的字冲掉
%saveas(gcf,picname)
saveas(gcf,picname)
end



plot(aaaa(2:40))
hold on
plot(temp1q(2:40))
hold on
plot(temp5q(2:40))