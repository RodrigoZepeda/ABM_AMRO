%File that creates csv files from the data to be read in Python
clear all
close all

load('data/NY_Network_1.mat');

%We go through each variable and export it to txt

%Variables
%------------------------------------------------------------
NPat    = Network.NPat;
writematrix(NPat,'data_to_py/NY_Network_1_NPat.txt')

NWard   = Network.NWard;
writematrix(NWard,'data_to_py/NY_Network_1_NWard.txt')

NDays   = Network.NDays; 
writematrix(NDays,'data_to_py/NY_Network_1_NDays.txt')

day0    = string(Network.day0);
writematrix(day0,'data_to_py/NY_Network_1_day0.txt')

WardSize = Network.WardSize;
writematrix(WardSize,'data_to_py/NY_Network_1_WardSize.txt')


%Tables
%------------------------------------------------------------

%Day position
daypos  = Network.daypos;
writetable(daypos,'data_to_py/NY_Network_1_daypos.csv') 

%Ward position
wardpos = Network.wardpos;
writetable(wardpos,'data_to_py/NY_Network_1_wardpos.csv') 

%Days
%------------------------------------------------------------
for d = 1:size(Network.Days,2)
    
    %These tables always have filling
    writetable(Network.Days(d).patients,strcat('data_to_py/NY_Network_1_patients_',string(d),'.csv')) 
    writetable(Network.Days(d).wards,strcat('data_to_py/NY_Network_1_wards_',string(d),'.csv')) 
    
    %This is a double
    writematrix(Network.Days(d).Pfirst,strcat('data_to_py/NY_Network_1_Pfirst_',string(d),'.csv')) 
    
    %Wfirst and activated might be empty so we skip those
    if ~isempty(Network.Days(d).Wfirst)
        writematrix(Network.Days(d).Wfirst,strcat('data_to_py/NY_Network_1_Wfirst_',string(d),'.txt')) 
    end
    
    if ~isempty(Network.Days(d).activated)
        writematrix(Network.Days(d).activated,strcat('data_to_py/NY_Network_1_activated_',string(d),'.csv')) 
    end
end



