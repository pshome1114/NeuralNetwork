function DeepLearningImageClassificationExampleTest
%Download the compressed data set from the following location
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
%Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder
if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
    % OR unzip(url, outputFolder);
end
outputFolder = uigetdir('', 'Select the folder with all image categories for the training set');
rootFolder = fullfile(outputFolder);
categories = {'Biomph', 'Bulinid','Lymnaea'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)

% We can activate this portion of code if we are interested in creating
% reflections of the minority data type. While this allows us to include
% more data of majority types it also overfits data of the minority type.
%myDir = uigetdir('', 'Select the folder with the smaller amount of images to augment the minority data set'); 
%myFiles = dir(fullfile(myDir,'*.jpg'));
%for k = 1:length(myFiles)
 %   baseFileName = myFiles(k).name;
 %   fullFileName = fullfile(myDir, baseFileName);
 %   fprintf(1, 'Now making a reflected image for %s\n', fullFileName);
 %   newImage = imread(fullfile(fullFileName));
 %   newImage = flipdim(newImage, 1);
 %   imwrite(newImage, fullfile(myDir,[baseFileName,'reflection.png']));
%end

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% Method 1: Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%Method 2: Create subsets with the number of subsets equal to the smallest number of
%images (minSetCount)
%largerGroupFolder = fullfile(rootFolder, 'airplanes');
%imds = imageDatastore(largerGroupFolder)
%nSets = minSetCounts;
%setSize = mod(max(tbl{:,2}), nSets);
%partition(imds, setSize);

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
% Find the first instance of an image for each category to ensure
% correct data input.
Biomphalaria = find(imds.Labels == 'Biomph', 1);
Bulinid = find(imds.Labels == 'Bulinid', 1);
Lymnaea = find(imds.Labels == 'Lymnaea', 1);

figure
subplot(1,3,1);
imshow(readimage(imds,Biomphalaria))
subplot(1,3,2);
imshow(readimage(imds,Bulinid))
subplot(1,3,3);
imshow(readimage(imds,Lymnaea))

% Load pre-trained VGG16 network
net = vgg19()
% View the CNN architecture
layers = net.Layers
% Inspect the first layer
net.Layers(1)
% Inspect the last layer
net.Layers(end)

% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
 function Iout = readAndPreprocessImage(filename)

        I = imread(filename);

        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(I)
            I = cat(3,I,I,I);
        end
        
        % Adjust size of the image 
        sz = net.Layers(1).InputSize; 
        Iout = I(1:sz(1),1:sz(2),1:sz(3));

        % Note that the aspect ratio is not preserved. In Caltech 101, the
        % object of interest is centered in the image and occupies a
        % majority of the image scene. Therefore, preserving the aspect
        % ratio is not critical. However, for other data sets, it may prove
        % beneficial to preserve the aspect ratio of the original image
        % when resizing.
 end
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

layersTransfer = net.Layers(1:end-3);
numClasses=(categories(trainingSet.Labels));
layersTransfer(end+1) = fullyConnectedLayer(3);
layersTransfer(end+1) = softmaxLayer();
layersTransfer(end+1) = classificationLayer();
optionsTransfer = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'InitialLearnRate',0.0001);
netTransfer = trainNetwork(trainingSet,layersTransfer,optionsTransfer);
YPred = classify(netTransfer,testSet);
YTest = testDigitData.Labels;

accuracy = sum(YPred==YTest)/numel(YTest)

convnetTransfer = trainNetwork(XTrain,TTrain, ...
		   layersTransfer,optionsTransfer);

% trainingFeatures = activations(net, trainingSet, featureLayer, ...
%     'MiniBatchSize', 32, 'channels', 'columns');
% % Get training labels from the trainingSet
% trainingLabels = trainingSet.Labels;

%optionsTransfer = trainingOptions('sgdm', ...
 %   'MaxEpochs',5, ...
 %  'InitialLearnRate',0.0001);


%Now we will loop over the images to make predictions
myDir = uigetdir('', 'Select the images to run predictions on'); %gets directory with images
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all jpg files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    newImage = fullfile(fullFileName);
    % Pre-process the images as required for the CNN
    img = readAndPreprocessImage(newImage);
    % Extract image features using the CNN
    % imageFeatures = activations(net, img, featureLayer);
    % Make a prediction using the classifier
    label = classify(net, img)
end
end
