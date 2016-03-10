function [class] = shapenetSynsetClass(synset)
%SHAPENETCLASSSYNSET Summary of this function goes here
%   Detailed explanation goes here
synsetNamePairs = {'02691156', 'aeroplane';
     '02834778', 'bicycle';
     '02858304', 'boat';
     '02876657', 'bottle';
     '02924116', 'bus';
     '02958343', 'car';
     '03001627', 'chair';
     '04379243', 'diningtable';
     '03790512', 'motorbike';
     '04256520', 'sofa';
     '04468005', 'train';
     '03211117', 'tvmonitor'};
 
class = synsetNamePairs(ismember(synsetNamePairs(:,1),synset),2);
if(~isempty(class))
    class = class{1};
end

end