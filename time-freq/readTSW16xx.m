%%% This script is used to read the binary file produced by the TSW1400
%%% and Radar Studio
%%% Command to run in Matlab GUI - readTSW16xx('<ADC capture bin file>')
function [retVal] = readTSW16xx(fileName)
%% global variables
% change based on sensor config
%----------------------------------------------------------------------
numADCSamples = 1024; % number of ADC samples per chirp
numADCBits = 16; % number of ADC bits per sample
numRX = 2; % number of receivers
numLanes = 2; % do not change. number of lanes is always 2
isReal = 0; % set to 1 if real only data, 0 if complex data0
%-----------------------------------------------------------------------
%% read file
% read .bin file
% fileName='C:\ti\mmwave_dfp_00_07_00_04\rf_eval\radarstudio\PostProc\数据\朝着雷达 (3).bin'
fid = fopen(fileName,'r');
adcData = fread(fid, 'uint16');
% compensate for offset binary format
adcData = adcData - 2^15;
% if 12 or 14 bits ADC per sample compensate for sign extension
% if numADCBits ~= 16
% l_max = 2^(numADCBits-1)-1;
% adcData(adcData > 1_max) = adcData(adcData > 1_max) - 2^numADCBits;
% end
fclose(fid);
% get total file
fileSize = size(adcData, 1);
test = adcData;
%% organize data by LVDS lane
% reshape data based on two samples per LVDS lane
adcData = reshape(adcData, numLanes*2, []);
% for real data
if isReal
% seperate each LVDS lane into rows
LVDS = zeros(2, length(adcData(1,:))*2);
% interleave the two sample sets from each lane
LVDS(1,1:2:end-1) = adcData(1,:);
LVDS(1,2:2:end) = adcData(2,:);
if numRX > 1
LVDS(2,1:2:end-1) = adcData(3,:);
LVDS(2,2:2:end) = adcData(4,:);
LVDS = reshape([LVDS(1,:);LVDS(2,:)], size(LVDS(1,:),1), []);
end
% for complex data
else
fileSize = fileSize/2;
% seperate each LVDS lane into rows
LVDS = zeros(2, length(adcData(1,:)));
% combine real and imaginary parts
LVDS(1,:) = adcData(1,:) + sqrt(-1)*adcData(2,:);
if numRX > 1
LVDS(2,:) = adcData(3,:) + sqrt(-1)*adcData(4,:);
LVDS = reshape([LVDS(1,:);LVDS(2,:)], size(LVDS(1,:),1), []);
end
end
%% organize data by receiver
% seperate each receiver into a single row
adcData = zeros(numRX, fileSize/numRX);
if numRX > 1
for j = 1:numRX
iter =1;
for i = (j-1)*numADCSamples+1:numADCSamples*numRX:fileSize
adcData(j,iter:iter+numADCSamples-1) = LVDS(1, i:i+numADCSamples-1);
iter = iter+numADCSamples;
end
end
else
adcData = LVDS(1,:);
end
%% return receiver data
retVal = adcData;
