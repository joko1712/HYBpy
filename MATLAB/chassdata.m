%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYNTHETIC DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nbatch, batch]=chassdata
rng default;

nbatch=5;
batchid=["chassbatch1","chassbatch2","chassbatch3","chassbatch4","chassbatch5"];

for i=1:nbatch
  
    data=readmatrix(batchid(i));
    
    t=data(:,1);
    
    species=data(:,2:12);
    
    species_true=data(:,2:12);
    
    vol=ones(length(t),1);
    
    raterule=[];
    
    sdspecies=repmat(max(species)*0.05,length(t),1);
    
    ualongtime=[];
    rvol_true=[];
    rann_true=[]; 
    
    batch(i).id=sprintf('batch%u',i);
    
    batch(i).u=[]; %controller parameters

    doplot=0;

    batch(i).np=           length(t);
    batch(i).t=            t;
    
    %species
    batch(i).cnoise=       species;
    batch(i).cnoise(1,:)=    species_true(1,:); %true initial values
 
    %compartment size
    batch(i).vol(:,1)=          vol;
    
    %raterules
    batch(i).raterule=     raterule;
    
    %state
    %batch(i).state =  [species, vol, raterule];
    batch(i).state =  [species, vol];
    batch(i).state(1,:)=    [species_true(1,:),vol(1)]; %true initial values
    %%%%%%%%%%%%%%%
   % batch(i).state =  [species_true, vol]; %TEST ONLY
   %%%%%%%%%%%%%%%%%
    batch(i).sc=      [sdspecies, ones(batch(i).np,1)];
    
    %control actions
    batch(i).ualongtime=   ualongtime;
    
    %true process data only important for plot purpose
    % if not available, assign null
    batch(i).c_true=       species_true;
    batch(i).rvol_true=    rvol_true;
    batch(i).rann_true=    rann_true;    

end

end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

