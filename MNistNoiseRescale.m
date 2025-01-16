%1.
%load images
sampletest = reshape(im2double(imread('sampletest.png')),784,1);
sampletrain = reshape(im2double(imread('sampletrain.png')),784,1);

%calculate most-likely estimate of 'a' for sample images using
%x = (x'*cov^-1*x)^-1*x'*cov^-1*x where cov = I from problem 2
aML = (sampletrain.'*sampletrain)^-1*sampletrain.'*sampletest;

%2.
%load in data
load('data.mat');

%calculate aML between training images, normalize by a, and calculate
%the euclidean distance.
%find the smallest and set that as the label for that image
classified = zeros(500,1);
totalErrorNum = 0;
for i = 1:500
    euclidians = zeros(5000,1);
    for j = 1:5000
        %reshape images
        testImage = reshape(imageTestNew(:,:,i),784,1);
        trainImage = reshape(imageTrain(:,:,j),784,1);
        %calculate MLE of a between 2 images
        astar = (trainImage.'*trainImage)^-1*trainImage.'*trainImage;
        %normalize test image (getes rid of amplification)
        normalTest = testImage / astar;
        %calculate euclidean distance
        euclideans(j) = norm(trainImage - normalTest);
    end
    %find min euclidean distance
    [val,minIndex] = min(euclideans);
    %classify test images
    classified(i) = labelTrain(minIndex);
    %calculate total number of error
    if classified(i) ~= labelTestNew(i)
        totalErrorNum = totalErrorNum + 1;
    end
end

%error
totalError = totalErrorNum/500;

errorPerClass = zeros(10,1);
for i = 1:10
    count = 0;
    errorCount = 0;
    for j = 1:500
        if labelTestNew(j) == i-1
            count = count + 1;
            if labelTestNew(j) ~= classified(j)
                errorCount = errorCount + 1;
            end
        end
    end
    errorPerClass(i) = errorCount/count;
end

%total error
totalErrorNum = totalErrorNum/500;

errorPerClass = zeros(10,1);
for i = 1:10
    count = 0;
    errorCount = 0;
    for j = 1:500
        if labelTestNew(j) == i-1
            count = count + 1;
            if labelTestNew(j) ~= classified(j)
                errorCount = errorCount + 1;
            end
        end
    end
    errorPerClass(i) = errorCount / count;
end

%plot error per class
figure;
bar([0,1,2,3,4,5,6,7,8,9],errorPerClass);
xlabel('Class')
ylabel('P(Error|Class)')
ylim([0,0.45]);

%nearest-neighbor classifier from HW2

dist = zeros(5000,500);
for i = 1:500
    for j = 1:5000
       diff = imageTrain(:,:,j) - imageTestNew(:,:,i);
       square = diff.^2;
       total = sum(sum(square));
       root = total^0.5;
       dist(j,i) = root;
    end
end

nnclass = zeros(500,1);
nnminindex = zeros(500,1);% contains index of min distance for each test image
for i=1:500
    x = find(dist(:,i) == min(dist(:,i)));
    nnminindex(i)=x;
    nnclass(i) = labelTrain(x);
end

NNPgC = zeros(1,10);
NNPE = 0;
for c = 0:9
    x = find(labelTestNew==c);
    total = length(x);
    nnerrorcount = 0;
    nntotalerrorcount = 0;
    for i=1:500
       if (labelTestNew(i)==c) && (nnclass(i) ~= labelTestNew(i))
           nnerrorcount = nnerrorcount + 1;
       end
       if (nnclass(i)~=labelTestNew(i))
           nntotalerrorcount = nntotalerrorcount + 1;
       end
    end
    NNPgC(c+1)=nnerrorcount/total;
    NNPE = nntotalerrorcount/500;
end

figure;
bar(0:9,NNPgC)
xlabel('NN Class')
ylabel('P(error|Class)')
%}
