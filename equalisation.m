% pre=processing video frames for map generation

% read images
original1 = imread("major1.jpg");
original2 = imread("major2.jpg");

% convert RGB image to HSV
temp1 = rgb2hsv(original1);
temp2 = rgb2hsv(original2);

% histogram equalisation on intensity component of image
eq1 = histeq(temp1(:,:,3));
eq2 = histeq(temp2(:,:,3));
temp1(:,:,3) = eq1;
temp2(:,:,3) = eq2;

% convert back to RGB format
final1 = hsv2rgb(temp1);
final2 = hsv2rgb(temp2);

% display
figure('Name','Histogram Equalisation');
hold on;
subplot(2,2,1);
imshow(original1);
title("Original Image 1");
subplot(2,2,2);
imshow(final1);
title("Equalised Image 1");
subplot(2,2,3);
imshow(original2);
title("Original Image 2");
subplot(2,2,4);
imshow(final2);
title("Equalised Image 2");