clc
clear

path = dir('images/test');
path = path(3:end);

rng(1024);
rand_idx = randperm(length(path));

for j = 1:200
    i = rand_idx(j);
    movefile(fullfile('images/test',path(i).name),fullfile('images/val',path(i).name));
end