clc
clear
%Add file path of the mat file for the tire run that you want 
load('C:\CODING\PYTHONCODING\Raw Tire DATA\R20 Runs\A2356raw12test.mat');

Mat1 = [ET,FX,FY,FZ,SA,P,TSTI,TSTC,TSTO,AMBTMP,IA];
%name the csv similarly to the original mat file as to make it easy to tell
%what run it is 
writematrix(Mat1,'C:\CODING\PYTHONCODING\SlicedRUNS\A2356FIXED12test.csv') 







