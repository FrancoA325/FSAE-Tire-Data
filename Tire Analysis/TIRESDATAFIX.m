clc
clear
%Add file path of the mat file for the tire run that you want 
load('C:\CODING\TireDataPY\Raw Tire DATA\LCO Runs\Cornering\A1965raw21.mat');

Mat1 = [ET,FX,FY,FZ,SA,P,TSTI,TSTC,TSTO,AMBTMP,IA];
%name the csv similarly to the original mat file as to make it easy to tell
%what run it is 
writematrix(Mat1,'C:\CODING\TireDataPY\SlicedRUNS\TESTout1.csv') 

%Drive Brake file conversion
load('C:\CODING\TireDataPY\Raw Tire DATA\LCO Runs\DriveBrake\A1464run30.mat');
Mat2 = [ET, FX, FZ, SA, P, TSTI, TSTC, TSTO, AMBTMP, IA, SL];
%name the csv similarly to the original mat file as to make it easy to tell
%what run it is 
writematrix(Mat2,'C:\CODING\TireDataPY\SlicedRUNS\TESTout2.csv') 





