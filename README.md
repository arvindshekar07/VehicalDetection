### **Vehicle Detection Project**

The goals / steps of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

[//]: # (Image References)
[image1]: output_images/hog.png 
[image2]: output_images/multiple_detection.png
[image3]: output_images/detection_heat_map.png
[image4]: output_images/HOG_expl1.png


###Introduction
Here  the aim of this  project is to detect vehical on the road by using computer
vision techniques . The code for this project is written in 
[VehicalDetection.ipynb file ](VehicalDetection.ipynb). Click on the link to check the code.

To give a gist of the project I needed to find all vehicle on a road. 
One approach was to use color threshold and do an image subtraction to check the movin object .
But this technique would fail if the was a vehicle that was moving in the same pace as your vehical 
and  to the computer vision techique this would appear as a static object.

One other technique is to use Histogram of Oriented Gradients  to find gradients in the color and spatial region for cars and non car objects and use that to classify car objects.
Initially I find these gradients initially in RGB color space and later found  YCrCb was a better choice in terms of the detections .


Based on these features I train a SVM classifier from those features set that is provided from a car and non car object.

Once the classifier is ready  this tested on a video where each frame is treated as an image
and use sliding window approach to move through the image to find a car like object.
 
The window is made in multiple sizes so as to capture car images that may apper 
close or away in a perspective images.

Based on this we get get a list of detection for the same car and unify this list of dection into once detection
and display as once  and draw a bounding box around it .


### Getting data
For this project i got most of the training data from  [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
I tested this pipline a video [here](project_video.mp4).

The data was initially accumlated into an array called cars and noncars.
```python
cars = []
for imtype in image_type:
    cars.extend(glob.glob(basedir+imtype+"/*"))

notcars = []
for imtype in image_type:
    notcars.extend(glob.glob(basedir+imtype+"/*"))

```
These two array will be used for training in an later stage. On downloading the data needed for training 
there are two folder called vehicals and non-vehicals.

###Histogram of Oriented Gradients (HOG)
As discussed earlier we get the HOG of an image based on color and spatial attributes and concatinate them both to get a 
good detection . With this the detection is much better.

To perform HOG orientations, pixels_per_cell and cells_per_block where chosen
```python

orient = 9
pix_per_cell =8
cell_per_block =2
hog_channel = 'ALL'# can be 0,1,2, ALL
```

The number of orientations is specified as an integer, and represents the number of orientation bins that the gradient information will be split up into in the histogram. Typical values are between 6 and 12 bins.

The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. This paramater is passed as a 2-tuple so you could have different cell sizes in x and y, but cells are commonly chosen to be square.

The cells_per_block parameter is also passed as a 2-tuple, and specifies the local area over which the histogram counts in a given cell will be normalized.

So I here I permitted 8 orents block and let 8 pixel per block. Here 'ALL' hog_channel refers to all the 
color channels to be used we are finding the features.

We can visualize the above concepts with this diagram below.

![image4]

I  convert the image to the color space of our interest .
```python
if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
            .
            .
            ...
           
```

Later I get get feature based on spatial and color  feature on the image.
```python
## getting the spatial features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    # return a stacked feature vector
    return np.hstack((color1, color2, color3))

##getting the color features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

```
The features coming out out of these two functions are concatinated  to form a single feature vector.

![image1]
The above every even images is an ouput that comes on performing HOG.
 
Once these features are collected for both cars and noncars they are combined to one single data set with
as X_series and all the car and non-car lable as 1 and 0 respectively in a y_series array

Both these are passed to SVM to predict cars
```python

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
svc.fit(X_train, y_train)
```
###Sliding Window Search

As mentioned above we apply a sliding window approach to check the presence of a car .
```python
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
overlap = 0.5
 windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(overlap,overlap))
   
```
As you can see form the code above the window size is about a 96x 96  and it would move to the next window by 0.5.
The y_start_stop =[400,656] tells the height at which the sliding should begin and end.
As sliding through all of the image is an expensive process we slide only bottom half of the image and thus we start from 400px to 656 px
![image2]

We can also apply thresholding and image heat/number detection of a single car detection
and consider it as a single detection.
![image3]

With thresholding technique we cam removes some of the false positives that get detected at times.

```python
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```

### Video Implementation
All the above procedure is combined into a single pipeline to find the vehicle.
```python

color_space = 'YCrCb'
orient = 9
pix_per_cell =8
cell_per_block =2
hog_channel = 'ALL'# can be 0,1,2, ALL
spatial_size=(32,32)#spatial binnig dimensions
hist_bins= 32
spatial_feat = True # spatical features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG  ffeatures on or off


ystart = 400
ystop = 656
scale = 1.5
d= deque(maxlen = 8)
def pipeline(img):
    
    out_img,heatmap = find_cars(img, ystart=ystart, ystop=ystop, 
                        scale=scale, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)
    
    
    d.append(heatmap)

    for i in range(len(d)):
        item = d[i]
        heatmap = cv2.add(heatmap,item)

    #     labels = label(heatmap)
    
    heatmap = apply_threshold(heatmap,6)

    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(img),labels)
    return draw_img


```
The scale choose is 1.5 . This valus is mostly through test and trial . I had initially tried with 1 and 1.3
and 1.5 did seem to do the work.
Here you can also see that a  queue of limited size has been implemeted 
to store all the consecutive frames heatmap  and add them all . Here I have kept the queue size as 8 .


You can see the output [here](project_video_result.mp4)
###Discussion

* This is was an intresting appraoach to find a car. But i feel that this approach may
have a back drop incase  the  another car  was right in front of another car.

* Also a case where two or more car have are of the same type and color  can also  bring forth a problem.

###References

* Udacity Self driving car project
* Stack overflow

