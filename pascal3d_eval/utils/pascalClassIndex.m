function [i] = pascalClassIndex(class,dataset)
%PASCALCLASSINDEX Summary of this function goes here
%   Detailed explanation goes here

if(nargin<2)
    dataset = '';
end

if(strcmp(dataset,'Ilsvrc'))
    globals;
    load(fullfile(ilsvrcDir,'classes'));
    i = find(strcmp(classes,class));
    return;
end

classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'};
i = find(ismember(classes,{class}));

end