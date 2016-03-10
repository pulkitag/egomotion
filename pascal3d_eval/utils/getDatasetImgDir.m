function [ imgDir,imgExt] = getDatasetImgDir(dataset)
%GETDATASETIMGDIR Summary of this function goes here
%   Detailed explanation goes here

globals;
switch dataset
    case 'pascal'
        imgDir = pascalImagesDir;
        imgExt = '.jpg';
        return
    case 'imagenet'
        imgDir = imagenetImagesDir;
        imgExt = '.JPEG';
        return
    case 'ilsvrc'
        imgDir = ilsvrcImagesDir;
        imgExt = '.JPEG';
        return

end

