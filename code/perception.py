import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to perform a color threshold for objective
def color_thresh_goal(img, rgb_thresh=(0, 0, 0)):
    # Create an empty array the same size in x and y as the image
    # but just a single channel
    color_select = np.zeros_like(img[:,:,0])
    # Apply the thresholds for RGB and assign 1's
    # where threshold was exceeded
    color_select = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Return the single-channel binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    # Define source and destination points
    source = np.float32([[13.6935, 139.248], [118.21, 95.3774], [198.855, 95.3774], [301.435, 139.248]])
    destination = np.float32([[155, 160-bottom_offset], [155, 150-bottom_offset], [165, 150-bottom_offset], [165, 160-bottom_offset]])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # Define color selection criteria
    ground_threshold = (200, 186, 165)
    target_threshold = (125, 100, 80)

    color_obs = np.ones_like(warped[:,:,0])

    ground = color_thresh(warped, rgb_thresh=ground_threshold)

    target = color_thresh_goal(warped, rgb_thresh=target_threshold)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = (255)*(color_obs - ground)
    Rover.vision_image[:,:,1] = (255)*target
    Rover.vision_image[:,:,2] = 255*ground

    # 5) Convert map image pixel values to rover-centric coords
    xpixO, ypixO = rover_coords(Rover.vision_image[:,:,0])
    xpixT, ypixT = rover_coords(Rover.vision_image[:,:,1])
    xpixG, ypixG = rover_coords(Rover.vision_image[:,:,2])

    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_x_world, obstacle_y_world = pix_to_world(xpixO, ypixO, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], dst_size*2)
    rock_x_world, rock_y_world = pix_to_world(xpixT, ypixT, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], dst_size*2)
    navigable_x_world, navigable_y_world = pix_to_world(xpixG, ypixG, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], dst_size*2)


    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 255
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 255
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 255

    # 8) Convert rover-centric pixel positions to polar coordinates
    #check for target rocks
    # Update Rover pixel distances and angles

    
    # REMOVED THIS BECAUSE ROCK TRACKING WAS UNSUCCESSFUL
    # if (np.sum(Rover.worldmap[:,:,1] > 10)):
    #     Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpixT, ypixT)
    #     Rover.mode = 'track'
    # else:
        
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpixG, ypixG)





    return Rover
