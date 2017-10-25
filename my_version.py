
def process_image(img):
    
    
#     dst_size = 5
#     bottom_offset = 6
    
#     source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
#     destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
#                   [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
#                   [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
#                   [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
#                   ])
    
    #NO MASK
    warped, mask = perspect_transform(img, source, destination)
    threshed = color_thresh(warped)
    obs_map = np.absolute(np.float32(threshed) - 1) * mask
    xpix, ypix = rover_coords(threshed)
    
    world_size = data.worldmap.shape[0]
    scale = 2 * dst_size
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]


    x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos,
                                   yaw, world_size, 2*dst_size)
    
    obsxpix, obsypix = rover_coords(obs_map)
   
    obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, xpos, ypos, 
                                            yaw, world_size, 2*dst_size)
    
    data.worldmap[y_world, x_world, 2] = 255
    data.worldmap[obs_y_world, obs_x_world, 0] = 255
    nav_pix = data.worldmap[:,:,2] > 0
    
#     data.worldmap[nav_pix, 0] = 0
    
    rock_map = find_rocks(warped, levels=(110, 110, 50))
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos,
                                                     ypos, yaw, world_size, scale)
        data.worldmap[rock_y_world, rock_x_world, 1] = 255
        
    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    #warped, mask = perspect_transform(img, source, destination)
    
    
    
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.4, 0)
    #cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.4, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
#     output_image[img.shape[0]:, 0:data.worldmap.shape[1], :] = np.flipud(map_add)
#     output_image[img.shape[0]:, 0:data.worldmap.shape[1], :] = data.worldmap
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)
    
#     print ('max val of the world map ', np.shape(data.worldmap))

        # Then putting some text over the image
#     cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
#                 cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image