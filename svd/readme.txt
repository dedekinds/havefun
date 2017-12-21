如果你是清华大学的学生，并且要完成李津老师高等数值分析的大作业，且苦于找不到
PROPACK包的话，那么恭喜你，辛苦了！

解压PROPACK.rar后，将解压后的文件夹放到MATLAB\R2017b\toolbox目录下，然后在MATLAB中输入pathtool，选择“添加并包含子文件夹”，
选择刚放进去的PROPACK文件夹，最后点保存，即可使用lansvd函数，

[U,S,V]=lansvd(A,k)，
默认情况下返回6个奇异值，k表示返回的奇异值个数O(∩_∩)O