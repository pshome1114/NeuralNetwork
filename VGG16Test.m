netData = load('convnet_checkpoint__50__2017_08_25__17_51_54.mat');
variablenames = fieldnames(netData);
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

for k = 1:numel(variablenames)
    net = netData.(variablenames{k});
    if isa(net, 'network')
        net = net;
        break;
    end
end


myDir = uigetdir('', 'Select the images to run predictions on'); %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
    img = imresize(img, [224 224]);
    % Extract image features using the CNN
    % imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = classify(net, img)
end
myDir = uigetdir('', 'Select the images to run predictions on'); %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
    img = imresize(img, [224 224]);
    % Extract image features using the CNN
    % imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = classify(net, img)
end
myDir = uigetdir('', 'Select the images to run predictions on'); %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
    img = imresize(img, [224 224]);
    % Extract image features using the CNN
    % imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = classify(net, img)
end