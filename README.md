# Vehicle Detection Project
### Josh Smith
#### 2017/03/19

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/HOG_car_ncar.png
[image2]: ./output_images/HOG_Param.png
[image3]: ./output_images/HOG_YCrCb.png
[image4]: ./output_images/color_spatial_BGR.png
[image12]: ./output_images/color_spatial_HLS.png
[image5]: ./output_images/sliding_window_search.png
[image6]: ./output_images/preprocess.png
[image7]: ./output_images/predictions.png
[image8]: ./output_images/window_heatmap_bbox.png
[image9]: ./output_images/heatmap_decay.png
[image10]: ./output_images/heatmap_history.png
[image11]: ./output_images/combine_heatmap.png
[video1]: ./output_videos/processed_project_video.mp4

### Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!  There is also great information in the [Jupyter notebook!](./Vehicle_Tracking.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The `generate_hog()` function extracts the HOG features without flattening them into the feature vector.  This allows the same function for both the training as well as the sliding window search.

![HOG Comparison][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

To explore the HOG features, I took a random car image and extracted a HOG visualization over a grid of orientations and cell sizes.  Visually it is becoming harder and harder to see the car image as the orientations decrease and the cell sizes increase.  There is a trade-off here as the parameter count increases rapidly with cell size reduction.  Orientations have a smaller impact on the total parameter count but increasing much above 5 it is difficult to see any improvement.  Training a classifier on a subset of the data to determine optimum parameters would be a good future exercise.  I settled on 9 orientations, 8 pixels per cell and a block size of 2.  This is an example of the HOG parameter grid for a car image:

![HOG Parameter Visualization][image2]

To explore the impact of the color space on the extracted HOG feature I flattened the extracted HOG map using `ravel()` and looked at the feature vector in a plot averaged over a set of car and non-car images.  I looked for colorspaces whose plots showed large separation between the car and non-car HOG features.  YCrCb and LAB both show good separation and were further explored in the classifier accuracy on the test data.

![HOG Colorspace][image3]

The HOG `transform_sqrt` parameter was tested on the test data accuracy and showed no improvement.

Final HOG parameters:
`color_space = YCrCb`
`orientations = 9`
`pixels_per_cell = 8`
`cells_per_block = (2, 2)`
`transform_sqrt = False`
`channels = [0, 1, 2]`

The HOG feature was extracted for all three color channels resulting in a feature size of 5292.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Three training features were used in the linear SVM: HOG, color histogram, and spatial histogram.  Each feature was explored separately to determine good starting parameters.
HOG was discussed in the previous section, so I'll skip to the color histogram.  The color histogram is simply a binning of color values for the image.  The spatial histogram is a downsampled and unraveled version of the image.  Each of these features utilized all their color channels.  To determine the best colorspace to use for the color and spatial histogram I examined the extracted features for a set of cars and non-cars in a plot.  To determine the best colorspace to use the plots were examined for separation between the car and non-car features.

![Color and Spatial Colorspace in BGR][image4]
![Color and Spatial Colorspace in HLS][image12]

The color histogram shows good distinction in the HLS colorspace while the spatial feature shows very good separation, surprisingly, in the RGB colorspace.  The spatial feature was minimized as much as possible while still achieving a remarkable feature.  An (8, 8) spatial size with all three channels was the final selection resulting in a feature size of 192.  The color histogram did not contribute as much to feature size but I did notice that as the color histogram count got large the more the classifier was unable to generalize.  32 bins with all three color channels for the color histogram size, this resulted in a feature size of 96.

##### Training the Classifier

Training  was done in the `train_classifier()` function.

The classifier was trained on all of the available car and non-car images provided as well as a few "trouble area" images extracted from the video.  The linear SVM was trained on 8794 car images and 8987 non-car images which is a balanced distribution.  

Image preprocessing was done in the `preprocess_image()` function.  This function converts the image to YUV and histogram equalized on the Y channel using the `cv2.equalizeHist()` function normalizing the brightness.  After equalization, the image was converted back to BGR colorspace and blurred using `cv2.GaussianBlur()` with a kernel size of (3,3) and sigma of 0.  After much experimentation this was a major improvement in the performance of the classifier in the test video, drastically reducing the number of false positives.  A sample of the process is shown below.

![Training equalization and blur][image6]

Each of the training images were preprocessed then the their feature vector was extracted.  The feature vectors for both car and non_car images were stacked together and a standard scalar was fit using `sklearn.preprocessing.StandardScaler` which normalizes the feature vectors.  The scalar was then applied to the feature vector to normalize and then split into training and test sets using the `split_data()` function.  Finally the linear SVM was trained on the training set then the trained classifier was evaluated on the test set for accuracy.  Training on the entire image set takes about 30 seconds and achieves an accuracy on the test set of 99.16%. The trained classifier is then stored in the class as well as saved to a pickle file for use later without the need to retrain.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was implemented in the `search_image()` function.

A sliding window search was used over four different scales.  The scaled search windows were squares of size 128, 106, 85 and 64.  Each search window was given a unique search area of the main image.  Each window was best at detecting cars of different sizes which correspond to different locations in the image.  The search areas were adjusted for each window size to give good detection results while restricting the search area to only that windows effective area.

![Sliding window sizes][image5]

The HOG extraction is a major source of computation time.  To try and minimize this as much as possible the image is restricted to just the search area for the given window size.  The HOG feature is extracted for this region of interest without flattening.  This HOG map is then subsampled for each sliding window then flattened to generate the feature vector.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

After training the classifier was tested on random sets of car and non-car images.

![Classifier testing][image7]

The training pipeline was then evaluated on the test images similar to the video.  Here you can see the sliding windows activated, the resulting heatmap and finally the cars identified and bounding boxes were drawn.

##### Full Image Pipeline Test
![Pipeline test][image8]

To optimize the performance of the classifier I tried to break down the classification features into the subparts and using simple graphical techniques I was able to quickly see what colorspaces and parameters to try training.  More on the parameter searching was discussed in the previous sections.  The trained classifier using the trial parameters was tested on the test images and test video to determine the performance.

Beyond the classifier, the sliding window search required a fair amount of optimization.  This was a balance of good car identification and computation time.  Search areas were selected to give good overlap between different search sizes.



---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/processed_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The first method to eliminate false positives was to increase the confidence threshold that what the classifier was detecting was actually a car.  This was done with `sklearn.svm.decision_function()` which returns the distance a given sample is from the decision boundary hyperplane.  This was used to filter out car detections with low confidence.

To filter false positives, I used a history of the generated heatmaps that were then summed, blurred and thresholded to generate a composite heatmap.  The history of heatmaps were progressively blurred after each frame such that the oldest heatmaps were dispersed and the recent heatmaps were sharp.  This decaying effect reduced the effects of small hot spots as well as allowed for motion frame to frame with better tracking.  This method provided good rejection of single frame false positives as well as give persistence to detected cars that are dropped from a frame.  This composite heatmap was then passed to the `scipy.ndimage.measurements.label()` to extract each of the remaining blobs.  These labels were assumed to be cars.  Bounding boxes are created from each of the labeled blobs and drawn onto the image.

### Here is the progression of a single heatmap as it decayed over time.

![Heatmap decay][image9]

### Here are the heatmaps that are combined:
![Heatmap history][image10]

The heatmaps above are added together, blurred and thresholded to generate the final heatmap for the current frame:

![Combined heatmap][image11]

---

### Discussion

#### 1. Briefly, discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Trying to eliminate false positives in the post processing proved to be challenging.  After several different methods to try and filter false positives from the heatmap, I moved my effort back to improving the classifier.  Using unique colorspaces that were better suited for each feature type proved to be a big improvement in accuracy.  The other large improvement was the addition of brightness equalization and slight blurring to the training set.  This was a reduction in classification accuracy on the test set, but it was a substantial improvement in the reduction of false positives in the video.  After the classifier was improved, the video processing was much easier.  I adopted the decaying heatmap technique that I started using while trying to eliminate false positives in my poor classifier.  This method has very nice stability and tracking while having a natural tendency to quickly reject false positives.  I think this same method could be applied without keeping some heatmaps in history and simply adding the new heatmap to a historic one that you decay each round.

While identifying cars is not an issue, keeping track of cars that are obscured by another car should be handled.  The traffic here is moving very slowly relative to the camera car.  I may see issues in tracking cars that are moving at a much greater relative speed.  Cars of different types would also pose a problem considering the training set.  I would expect issues with vehicles like semi-trucks, trucks with trailers or industrial equipment like garbage trucks, cranes, dump trucks, etc.

The classifier here runs at an alarmingly slow rate and is unacceptable for realtime tracking.  I would expect that there are more state-of-the-art techniques that don't require sliding window searches.  Using a larger training set with possibly more classifications would improve the detection of vehicles that are not passenger cars.  The prediction speed of a ConvNet would be interesting to compare to the feature extraction and prediction speed of the linear SVM.  
