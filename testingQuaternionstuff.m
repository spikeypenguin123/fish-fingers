 
clear all
close all
clc
% test 

% q = [1,0,0,0];
% q = [1,1,0,0];
q = [0.97547004, 0.02369198, 0.21203752, -0.05419399];


ZYX = quat2eul(q,'ZYX');
ZYZ = quat2eul(q,'ZYZ');
XYZ = quat2eul(q,'XYZ');

eulers = (q2e(q'))';  

% ZYX = quat2eul(q,'ZYX')*180/pi;
% ZYZ = quat2eul(q,'ZYZ')*180/pi;
% XYZ = quat2eul(q,'XYZ')*180/pi;
% 
% eulers = (q2e(q'))'*180/pi;  

% 
% 
% function eulers = q2e(q)   
%     % in: 4xn quaternions
%     % out: 3xn eulers (rads)
% 
% %     % normalise quaternions
% %     q = q./sqrt(sum(q.^2,1));
%     
%     q_0 = q(1,:);
%     q_1 = q(2,:);
%     q_2 = q(3,:);
%     q_3 = q(4,:);
%     
%     eulers = [
%         atan2(2.*(q_0.*q_1+q_2.*q_3),1-2.*(q_1.^2+q_2.^2));
%         asin(2.*(q_0.*q_2-q_3.*q_1));
%         atan2(2.*(q_0.*q_3+q_1.*q_2),1-2.*(q_2.^2+q_3.^2));
%     ];
% 
% end