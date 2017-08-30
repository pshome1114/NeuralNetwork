#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name  
#SBATCH --job-name=AlexNetTest
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=AlexNetTest.%j.out
#################
# a file for errors from the job
#SBATCH --error=AlexNetTest.%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,  
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition 
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM) 
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#################
#number of nodes you are requesting, the more you ask for the longer you wait
#SBATCH --nodes=1
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=4000
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=pshome@stanford.edu
#now run normal batch commands
# note the "CMD BATCH is an R specific command
module load matlab
matlab
# You can use srun if your job is parallel
# otherwise: 
function DeepLearningImageClassificationExampleTest
outputFolder = fullfile('/home', 'pshome', 'SNAIL_PICS');
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

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
% Find the first instance of an image for each category to ensure
% correct data input.
Biomphalaria = find(imds.Labels == 'Biomph', 1);
Bulinid = find(imds.Labels == 'Bulinid', 1);
Lymnaea = find(imds.Labels == 'Lymnaea', 1);

% figure
% subplot(1,3,1);
% imshow(readimage(imds,Biomphalaria))
% subplot(1,3,2);
% imshow(readimage(imds,Bulinid))
% subplot(1,3,3);
% imshow(readimage(imds,Lymnaea))

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
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
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
confMat = confusionmat(testLabels, predictedLabels)

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))

%Now we will loop over the images to make predictions
myDir = fullfile('/home', 'pshome', 'SNAIL_PICS','Biomph'); %gets directory with images
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
myDir = fullfile('/home', 'pshome', 'SNAIL_PICS','Bulinid'); %gets directory with images
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
myDir = fullfile('/home', 'pshome', 'SNAIL_PICS','Lymnaea'); %gets directory with images
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