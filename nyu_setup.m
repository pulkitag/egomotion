%dat = load('/data0/pulkitag/data_sets/nyu2/nyu_depth_v2_labeled.mat');
pth = fullfile('/data0/pulkitag/data_sets/nyu2/ims/im%04d.jpg');
images = dat.images;
[H,W,ch,N] = size(images);
for i = 1:1:N
	im = images(:,:,:,i);
	imwrite(im, sprintf(pth, i));	
end
