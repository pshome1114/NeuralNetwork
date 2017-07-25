function DeepLearningImageClassificationExample
% Download the compressed data set from the following location
%url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
%outputFolder = fullfile(tempdir, 'caltech101'); % define output folder
%if ~exist(outputFolder, 'dir') % download only once
%    disp('Downloading 126MB Caltech101 data set...');
%    untar(url, outputFolder);
%    %unzip(url, outputFolder);
%end
rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'starfish', 'water_lilly', 'sunflower'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%Method 1: Create subsets with the number of subsets equal to the smallest number of
%images (minSetCount)
%largerGroupFolder = fullfile(rootFolder, 'airplanes');
%imds = imageDatastore(largerGroupFolder)
%nSets = minSetCounts;
%setSize = mod(max(tbl{:,2}), nSets);
%partition(imds, setSize);

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
% Find the first instance of an image for each category
starfish = find(imds.Labels == 'starfish', 1);
water_lilly = find(imds.Labels == 'water_lilly', 1);
sunflower = find(imds.Labels == 'sunflower', 1);

figure
subplot(1,3,1);
imshow(readimage(imds,starfish))
subplot(1,3,2);
imshow(readimage(imds,water_lilly))
subplot(1,3,3);
imshow(readimage(imds,sunflower))
% Load pre-trained AlexNet
net = alexnet()
% View the CNN architecture
net.Layers
% Inspect the first layer
net.Layers(1)
% Inspect the last layer
net.Layers(end)

% Number of class names for ImageNet classification task
numel(net.Layers(end).ClassNames)
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
 function Iout = readAndPreprocessImage(filename)

        I = imread(filename);

        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(I)
            I = cat(3,I,I,I);
        end

        % Resize the image as required for the CNN.
        Iout = imresize(I, [227 227]);

        % Note that the aspect ratio is not preserved. In Caltech 101, the
        % object of interest is centered in the image and occupies a
        % majority of the image scene. Therefore, preserving the aspect
        % ratio is not critical. However, for other data sets, it may prove
        % beneficial to preserve the aspect ratio of the original image
        % when resizing.
 end
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')
featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% Extract test features using the CNN
testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))

%Now we will loop over the images to make predictions
myDir = uigetdir; %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
    img = readAndPreprocessImage(newImage);
    % Extract image features using the CNN
    imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures)
end
end
