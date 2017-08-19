## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./images/perception.png
[image2]: ./images/perspective.png
[image3]: ./images/warp_perspective.png
[image4]: ./images/map_nav_angles.png
[image5]: ./images/rover_centric.png
[image6]: ./images/map_to_world.png
[image7]: ./images/decision.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

Results from data I created personally:


#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 




### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

The perception step came down to finding the right color thersholds to use to distinguish the three types (navaigable terrain, obstacle, target):

![alt text][image1]

The next step was to complete a perspective transform which converts the rover camera input:

![alt text][image2]

into a "satellite" view:

![alt text][image3]

Then I combined the thresholding into the warped perspective to create navigation angles:

![alt text][image4]

and then converted them to rover centric angles:

![alt text][image5]

and finalled mapped the findings to the world map:

![alt text][image6]

Finally, the summary of the steps is below:

![alt text][image7]

Start:

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

I ran the simulation on a Windows 10 desktop computer 1280x720 with Fantastic graphic settings.

I was very succesful in transforming the rover camera image into navigable angles for the rover to drive in. I did however struggle in creating logic to guide the rover towards the target rocks. I attempted to create a new mode called "tracking" (I commented the code out in my submission) in order to slow the rover and guide it towards the rocks. I had many complications where the rover would simply stop and spin and was unable to teach it to pick up rocks. I will keep working towards a more effective solution in the weeks to come, I found it to be an interesting challenge.





