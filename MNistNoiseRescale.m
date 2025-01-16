%Step 1: Load the images and labels, create placeholder variables
load("data.mat");
%load("label.mat");
distMat = zeros(28,28);
labelResults = zeros(500,1);
closestImg = zeros(500,1);
errorNums = zeros(10,1); %Store number of errors
correctNums = zeros(10,1); %Store number correct
errorCalc = zeros(10,1); %Store error percent values
errorImages = zeros(5,1); %Store the first 5 error image values
double distance;
double euclid;
double bestAvg;
double labelVal;
double totalError;
double errorImVal;
double bestDist;
double a_P1; %This variable is the value of a for part 1
double a_im; %This is the value of a for each image
double lSquares; %This is used for calculating the least squares distance
labelVal = 0;
distance = 0;
a_P1 = 0;
a_im = 0;
bestDist = 200000; %initiate with highest possible pixel value
errorImVal = 1; %Use to count number of error images
%Now load images for part 1, convert to doubles, and reshape
% In order to receive 1 value for a
sampleTest = imread("sampletest.png");
sampleTrain = imread("sampletrain.png");
sampleTest = im2double(sampleTrain);
sampleTrain = im2double(sampleTest);
sampleTest = reshape(sampleTest, 784, 1);
sampleTrain = reshape(sampleTrain, 784, 1);
%Problem 1: Calculate the ML estimate of parameter 'a'
% Θ=[(x^T)x]^-1 * x^T * y where x is train image y is test image
a_P1 = inv(transpose(sampleTrain) * sampleTrain) * transpose(sampleTrain) * sampleTest;
%Step 2: Distance the test image values at each pixel to the values
       % of all the pixels in the training images
       % e.g. imageTest(1,20,4) vs imageTrain(1,20, i) 1 <= i <= 5000
%For this assignment, we'll be adding a step where we find the 'a' for each
%image pair as well
for i = 1:500 %Loops through the test images
   for j = 1:5000 %Loops through the training images
       %for u = 1:784 %Loops through the rows
           %for v = 1:28 %Loops through the columns
               %reshape the images to find a
               imageTest = reshape(imageTestNew(:,:,i), 784, 1);
               imageTrainee = reshape(imageTrain(:,:,j), 784, 1);
               a_im = inv(transpose(imageTrainee) * imageTrainee) * transpose(imageTrainee) * imageTest;
              
               for u = 1:784
                   %Begin least squares (y/a - x)^2
                   lSquares = imageTest(u,1) / a_im;
                   lSquares = (lSquares - imageTrainee(u,1))^2;
                   distance = lSquares + distance;
               end
               %Implement Euclidean distance
               %euclid = imageTest(u,v,i) - imageTrain(u,v,j);
               %euclid = euclid ^ 2;
               %distance = distance + euclid;
           %end
       %end
       distance = sqrt(distance); %Taking sqrt of the sum of dist squares
       if distance < bestDist %New average value is lower than best
           labelResults(i,1) = labelTrain(j,1); %Copy label from test img
           closestImg(i,1) = j; %Store NN img # for part 3
           bestDist = distance; %Store new record low avg #
       end
      
       distance = 0; %Reset the square distance counter for next train image
   end
   bestDist = 200000; %Reset best average counter for new test image
end
%Step 4: Once every image is tested, find error count
for i = 1:500
   if labelResults(i,1) == labelTestNew(i,1) %If correct guess
       labelVal = labelTestNew(i,1) + 1; %Matrices start at 1, not 0
       correctNums(labelVal,1) = correctNums(labelVal,1) + 1;
   end
   if labelResults(i,1) ~= labelTestNew(i,1) %If incorrect guess
       if errorImVal <= 5 %Grab 5 error images for comparison
           errorImages(errorImVal,1) = i;
           errorImVal = errorImVal + 1;
       end
       labelVal = labelTestNew(i,1) + 1;  %Offput because matrix starts at 1
       errorNums(labelVal,1) = errorNums(labelVal,1) + 1;
   end
end
%Step 5: For problem 1, plot error rates by image class
for i = 1:10 %Add to find the total number of images per image class
   errorCalc(i,1) = correctNums(i,1) + errorNums(i,1);
end
for i = 1:10 %Divide number of errors by total
   errorCalc(i,1) = errorNums(i,1) / errorCalc(i,1);
end
x = 0:1:9; %Specifies x values for graph due to matrix offput by 1
bar(x,errorCalc) %Creates the bar graph
disp(errorCalc) %Prints the values of errorCalc
%Step 6: For problem 2, find total error rate
totalError = sum(errorCalc); %Add all error values together
totalError = totalError/10; %Divide them by 10, the number of digits
disp(totalError); %Prints the value totalError


Code for Problem 3 (modified code for euclidean distance):


%Step 1: Load the images and labels, create placeholder variables
load("data.mat");
%load("label.mat");
distMat = zeros(28,28);
labelResults = zeros(500,1);
closestImg = zeros(500,1);
errorNums = zeros(10,1); %Store number of errors
correctNums = zeros(10,1); %Store number correct
errorCalc = zeros(10,1); %Store error percent values
errorImages = zeros(5,1); %Store the first 5 error image values
double distance;
double euclid;
double bestAvg;
double labelVal;
double totalError;
double errorImVal;
double bestDist;
labelVal = 0;
distance = 0;
bestDist = 200000; %initiate with highest possible pixel value
errorImVal = 1; %Use to count number of error images
%Step 2: Distance the test image values at each pixel to the values
       % of all the pixels in the training images
       % e.g. imageTest(1,20,4) vs imageTrain(1,20, i) 1 <= i <= 5000
for i = 1:500 %Loops through the test images
   for j = 1:5000 %Loops through the training images
       for u = 1:28 %Loops through the rows
           for v = 1:28 %Loops through the columns
               %Implement Euclidean distance
               euclid = imageTestNew(u,v,i) - imageTrain(u,v,j);
               euclid = euclid ^ 2;
               distance = distance + euclid;
           end
       end
       distance = sqrt(distance); %Taking sqrt of the total distance squares
       if distance < bestDist %New average value is lower than best
           labelResults(i,1) = labelTrain(j,1); %Copy label from test img
           closestImg(i,1) = j; %Store NN img # for part 3
           bestDist = distance; %Store new record low avg #
       end
      
       distance = 0; %Reset the square distance counter for next train image
   end
   bestDist = 200000; %Reset best average counter for new test image
end
%Step 4: Once every image is tested, find error count
for i = 1:500
   if labelResults(i,1) == labelTestNew(i,1) %If correct guess
       labelVal = labelTestNew(i,1) + 1; %Matrices start at 1, not 0
       correctNums(labelVal,1) = correctNums(labelVal,1) + 1;
   end
   if labelResults(i,1) ~= labelTestNew(i,1) %If incorrect guess
       if errorImVal <= 5 %Grab 5 error images for comparison
           errorImages(errorImVal,1) = i;
           errorImVal = errorImVal + 1;
       end
       labelVal = labelTestNew(i,1) + 1;  %Offput because matrix starts at 1
       errorNums(labelVal,1) = errorNums(labelVal,1) + 1;
   end
end
%Step 5: For problem 1, plot error rates by image class
for i = 1:10 %Add to find the total number of images per image class
   errorCalc(i,1) = correctNums(i,1) + errorNums(i,1);
end
for i = 1:10 %Divide number of errors by total
   errorCalc(i,1) = errorNums(i,1) / errorCalc(i,1);
end
x = 0:1:9; %Specifies x values for graph due to matrix offput by 1
bar(x,errorCalc) %Creates the bar graph
disp(errorCalc); %Prints the values of errorCalc
%Step 6: For problem 2, find total error rate
totalError = sum(errorCalc); %Add all error values together
totalError = totalError/10; %Divide them by 10, the number of digits
disp(totalError); %Prints the value totalError
