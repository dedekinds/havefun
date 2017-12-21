% 
% % temp=diag([[30:-1:1],0.00001.*ones(1,247)])
% check1=[100,0.0001.*ones(1,99)];
% check2=[[100:-1:91],0.0001.*ones(1,90)];
% check3=[100:-1:1];
% check4=[10000,0.0001.*ones(1,99)];
% check5=0.0001.*ones(1,100);
% check6=[[100:-1:91],0.0001.*ones(1,20),0.*ones(1,70)];
% check7=[0.0001.*ones(1,20),zeros(1,80)];
% I=imread('temp5.jpg');
% R=double(I(:,:,1));
% [a,b,c]=svd(R);
% b2=diag(b);
% check8=b2(1:100)';
% check9=100.*rand(1,100);
% check10=100.*randn(1,100);
% 
% R=100.*randn(100,100);
% [U,s,V]=svd(R);

temp=check4;
test=U*diag(temp)*V';
tic
[Ru,Rz,Rv]=svd(test);
toc

tic
[u,z,v]=lansvd(test,100);
toc

error1_qyz=abs(sum(sum(diag(temp).^2-Rz.^2)))
error2_qyz=abs(sum(sum(diag(temp).^2-z.^2)))

error1_zuo=abs(sum(sum(U.^2-Ru.^2)))
error2_zuo=abs(sum(sum(U.^2-u.^2)))


error1_you=abs(sum(sum(V.^2-Rv.^2)))
error2_you=abs(sum(sum(V.^2-v.^2)))




res=[0.163	0.239	1.8238e-14	1.4211e-14	1.4540e-14	2.1677e-14	1.6196e-14	9.2868e-15
0.171	0.245	1.1369e-13	1.8474e-13	3.8891e-15	2.1389e-14	1.4142e-14	1.1962e-14
0.162	0.308	1.9895e-13	1.4211e-13	1.1561e-14	1.7966e-14	5.2035e-15	1.2748e-14
0.160	0.231	7.4506e-14	1.0431e-13	2.6193e-14	2.2629e-14	2.0122e-14	1.1882e-14
0.164	0.278	2.1684e-19	8.1315e-20	1.8713e-14	1.9909e-14	2.3922e-14	1.2431e-14
0.160	0.252	1.9895e-13	9.9476e-14	2.0852e-14	2.2478e-14	1.3492e-14	1.1899e-14
0.160	0.221	1.0842e-19	8.1315e-20	4.1449e-14	2.0717e-14	1.6858e-14	9.5628e-15
0.163	0.201	3.0923e-11	2.5466e-11	3.0300e-14	1.9405e-14	5.9694e-14	1.1775e-14
0.161	0.219	1.7792e-10	2.5807e-11	8.4480e-15	1.8579e-14	6.8098e-15	1.2419e-14
0.159	0.195	2.2737e-11	3.4288e-10	1.4453e-14	1.8216e-14	1.3964e-14	1.4236e-14
]
plot(res(:,5))
hold on 
plot(res(:,6))