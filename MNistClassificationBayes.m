%1.
%calculate sampleMean of each class
sampleMeans = zeros(784,10);
for i = 1:10
    sum = zeros(784,1);
    num = 0;
    for j = 1:5000
        if labelTrain(j) == (i-1)
            sum = sum + reshape(imageTrain(:,:,j),784,1);
            num = num + 1;
        end
        sampleMeans(:,i) = sum./num;
    end
end
%plot sampleMeans
for i = 1:10
    subplot(2,5,i)
    imshow(reshape(sampleMeans(:,i),28,28),[])
end

%2.
%Classification by BDR
testClassification = zeros(500,1); % contains classified labels
for i = 1:500
    classProb = zeros(10,1);
    for j = 1:10
        %note inverse of identity covariance is 1
        %can ignore right-hand side of equation since it is constant
        %right hand side looks as follows:
        %constants = -0.5*log(2*pi)^28^2 + log(1/10);
        classProb(j) = -0.5*(reshape(imageTest(:,:,i),784,1)-sampleMeans(:,j))'*(reshape(imageTest(:,:,i),784,1)-sampleMeans(:,j));
    end
    [Y,I] = max(classProb); %Y = value, I = index
    testClassification(i) = I-1;
end      
%find error
countTrain = zeros(1,10);
numError = zeros(1,10);
for i = 1:10
    for j = 1:500
        if labelTest(j) == i-1
            countTrain(i) = countTrain(i) + 1;
            if testClassification(j) ~= labelTest(j)
                numError(i) = numError(i) + 1;
            end
        end
    end
end
probErrorClass = zeros(1,10);
for i = 1:10
    probErrorClass(i) = numError(i)/countTrain(i);
end      
%plot error per class
figure;
x = 0:9;
bar(x,probErrorClass);
%calculate total error probability
totalNumError = 0;
for i = 1:10
    totalNumError = totalNumError + numError(i);
end
totalProbError = totalNumError/500

%3.
%calculate covariance a class at at time
covs = zeros(784,784,10);
for i=1:10
    collect = zeros(1,784);
    count = 0;
    for j = 1:5000
        if(labelTrain(j)==i-1)
            collect = [collect;reshape(imageTrain(:,:,j),784,1)'];
            count = count + 1;
        end
        
    end
    covs(:,:,i) = cov(collect(2:end,:));
end
%diplay covariance
figure;
for i=1:10
    subplot(2,5,i)
    %imagesc(covariances(:,:,1))
    imshow(covs(:,:,i),[])
end
