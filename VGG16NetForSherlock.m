outputFolder = fullfile('/home', 'pshome', 'SNAIL_PICS');
rootFolder = fullfile(outputFolder);
categories = {'Biomph', 'Bulinid','Lymnaea'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% Method 1: Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');



% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
% Find the first instance of an image for each category to ensure
% correct data input.

load vgg16.mat;
% Load pre-trained VGG16 network
net = vgg16()
% View the CNN architecture
layers = net.Layers
% Inspect the first layer
net.Layers(1)
% Inspect the last layer
net.Layers(end)

% Set the ImageDatastore ReadFcn
 
imds.ReadFcn = @(filename)imresize((imread(filename)), [224 224]);
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

layersTransfer = net.Layers(1:end-3);
categories(trainingSet.Labels)
numClasses=numel(unique(categories(trainingSet.Labels)))
layersTransfer(end+1) = fullyConnectedLayer(numClasses, 'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);
layersTransfer(end+1) = softmaxLayer();
layersTransfer(end+1) = classificationLayer();
optionsTransfer = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'InitialLearnRate',0.0001);
netTransfer = trainNetwork(trainingSet,layersTransfer,optionsTransfer);
YPred = classify(netTransfer,testSet);
YTest = testSet.Labels;

accuracy = sum(YPred==YTest)/numel(YTest)

%Now we will loop over the images to make predictions
myDir = fullfile('/home', 'pshome', 'SNAIL_PICS','Biomph'); %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
    I = imread(newImage);
    % Resize the image as required for the CNN.
    img = imresize(I, [227 227]);
    % Extract image features using the CNN
    imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures)
end
myDir = fullfile('/home', 'pshome', 'SNAIL_PICS','Bulinid'); %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
        I = imread(newImage);
        % Resize the image as required for the CNN.
        img = imresize(I, [227 227]);
    % Extract image features using the CNN
    imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures)
end
myDir = fullfile('/home', 'pshome', 'SNAIL_PICS','Lymnaea'); %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
        I = imread(newImage);
        % Resize the image as required for the CNN.
        img = imresize(I, [227 227]);
    % Extract image features using the CNN
    imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures)

