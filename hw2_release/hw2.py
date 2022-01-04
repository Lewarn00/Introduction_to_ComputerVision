import numpy as np
from canny import *

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 7.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points
      mask        - (optional, for your use only) foreground mask constraining
                    the regions to extract interest points
   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points = 200, scale = 1.0, mask = None):
  # check that image is grayscale
  assert image.ndim == 2, 'image should be grayscale'
  ##########################################################################
  masked = image[:]

  if not mask is None:
    masky, maskx = mask.shape
    for y in range(masky):
      for x in range(maskx):
        if mask[y,x] == False:
          masked[y,x] = 0

  scores = []
  xs = []
  ys = []

  image_gauss = conv_2d_gaussian(masked, sigma = 1.0)
  output_img = np.zeros_like(image_gauss)

  window_size = scale * 3 
  k = 0.06
  offset = int(window_size/2)
  y_range = image_gauss.shape[0] - offset
  x_range = image_gauss.shape[1] - offset
    
  dx, dy = sobel_gradients(image_gauss)
  Ixx = conv_2d_gaussian(dx**2, sigma = 1.0) 
  Ixy = conv_2d_gaussian(dy*dx, sigma = 1.0)
  Iyy = conv_2d_gaussian(dy**2, sigma = 1.0) 

  output = []
  for y in range(y_range):
    output.append([])
  temp_output = []
  for y in range(offset, y_range):
    for x in range(offset, x_range):
      starty = y - offset
      endy = y + offset + 1
      startx = x - offset
      endx = x + offset + 1
      windowed_Ixx = Ixx[starty:endy, startx:endx]
      windowed_Ixy = Ixy[starty:endy, startx:endx]
      windowed_Iyy = Iyy[starty:endy, startx:endx]
      Sxx = windowed_Ixx.sum()
      Sxy = windowed_Ixy.sum()
      Syy = windowed_Iyy.sum()
      det = (Sxx * Syy) - (Sxy**2)
      trace = Sxx + Syy
            
      r = det - k*(trace**2)
      output[y].append(r)
      temp_output.append([x,y,r])

  threshold_temp = sorted(temp_output, key=lambda x: x[2])[-int(max_points*4):] 
  temp_xs = [x[0] for x in threshold_temp]
  temp_ys = [x[1] for x in threshold_temp]

  final_scores = []
  max_offset = int(window_size)
  numpy_output = np.array(output[max_offset:])
  for y in range(max_offset, y_range - max_offset):
    for x in range(max_offset, x_range - max_offset):
      if y in temp_ys and x in temp_xs:
        starty = y - max_offset
        endy = y + max_offset + 1
        startx = x - max_offset
        endx = x + max_offset + 1
        max_window = numpy_output[starty : endy, startx : endx]
        window_max = max([max(x) for x in max_window])
        if output[y][x] >= window_max:
          final_scores.append([x,y,output[y][x]])
  
  txs = [x[0] for x in final_scores]
  for t in range(len(temp_output)):
    if temp_output[t][0] not in txs and temp_output[t][0] in temp_xs:
      temp_output[t][2] = 0

  revised_threshold = sorted(temp_output, key=lambda x: x[2])[-int(max_points*4):]
  revised_temp_xs = [x[0] for x in revised_threshold]
  revised_temp_ys = [x[1] for x in revised_threshold]

  max_offset = int(window_size)
  final_final_scores = []
  for y in range(max_offset, y_range - max_offset):
    for x in range(max_offset, x_range - max_offset):
      if y in revised_temp_ys and x in revised_temp_xs:
        starty = y - max_offset
        endy = y + max_offset + 1
        startx = x - max_offset
        endx = x + max_offset + 1
        max_window = numpy_output[starty : endy, startx : endx]
        window_max = max([max(x) for x in max_window])
        if output[y][x] >= window_max:
          final_final_scores.append([x,y,output[y][x]])

  threshold = sorted(final_final_scores, key=lambda x: x[2])[-int(max_points * 0.9):] 

  xs = [x[0] for x in threshold]
  ys = [y[1] for y in threshold]
  scores = [s[2] for s in threshold]  
  ##########################################################################
  return xs, ys, scores

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   width_x, width_y = 15, 15
   image_border = mirror_border(image, width_x, width_y)
   k = []
   mag, theta = canny_nmax(image_border)

   for y,x in zip(xs,ys):
    x += width_x
    y += width_y
    magwindow = mag[x-width_x:x+width_x, y-width_y:y+width_y]
    thetawindow = theta[x-width_x:x+width_x, y-width_y:y+width_y]
    k2 = []
    for i in range(1,4):
      for j in range(1,4):
        window_theta = thetawindow[(i-1)*10:i*10, (j-1)*10:j*10]
        window_mag = magwindow[(i-1)*10:i*10, (j-1)*10:j*10]
        hist, bins = np.histogram(window_theta, bins = 8, range = (-np.pi,np.pi), weights = window_mag)
        k2.append(hist)
    k.append([item for sublist in k2 for item in sublist])
    f = np.array(k)
    norm = np.linalg.norm(f)
    if norm == 0:
      feats = f
    else:
      feats = f / norm 
   ##########################################################################
   return np.array(feats)

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())

   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""
def match_features(feats0, feats1, scores0, scores1):
   ##########################################################################
  f0_range, k = np.shape(feats0)
  f1_range = np.shape(feats1)[0]
  matches = np.zeros(f0_range, dtype=np.int8)
  scores = np.zeros(f0_range)

  for i in range(f0_range):
    distances = np.zeros(f1_range)
    rem = 0
    for j in range(f1_range):
      sums = 0
      for n in range(k):
        sums += (feats0[i,n] - feats1[j,n]) ** 2
      dist = np.sqrt(sums)
      distances[j] = dist  
      rem = abs(scores0[i] - scores1[j]) #not used   

    sorted_index = np.argsort(distances)
    sorted_dist = np.sort(distances) 

    NN_Ratio = sorted_dist[0]/sorted_dist[1]
    
    for m in range(len(matches)):
      if sorted_index[0] == matches[m]:
        if NN_Ratio < scores[m]:
          scores[i] = 0 
        elif NN_Ratio > scores[m]:
          scores[i] = NN_Ratio
          scores[m] = 0
        elif NN_Ratio == scores[m]:
          scores[m] = 0
          scores[i] = 0
    
    matches[i] = sorted_index[0]
    scores[i] = NN_Ratio 
   ##########################################################################
  return matches, scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################
  differences1 = []
  for i in range(len(xs0)):
    y1 = xs1[matches[i]]
    x1 = xs0[i]
    differences1.append(int(y1 - x1))

  differences2 = []
  for j in range(len(ys0)):
    y2 = ys1[matches[j]]
    x2 = ys0[j]
    differences2.append(int(y2 - x2))

  hist1, bins1 = np.histogram(np.array(differences1), bins = 50, range = (min(differences1),max(differences1)), weights = scores)
  tx = bins1[np.argmax(hist1)] 

  hist2, bins2 = np.histogram(np.array(differences2), bins = 50, range = (min(differences2),max(differences2)), weights = scores)
  ty = bins2[np.argmax(hist2)] 

  votes = [max(hist1), max(hist2)]
  print(votes)
  ##########################################################################
  return tx, ty, votes

"""
    OBJECT DETECTION (10 Points Implementation + 5 Points Write-up)

    Implement an object detection system which, given multiple object
    templates, localizes the object in the input (test) image by feature
    matching and hough voting.

    The first step is to match features between template images and test image.
    To prevent noisy matching from background, the template features should
    only be extracted from foreground regions.  The dense point-wise matching
    is then used to compute a bounding box by hough voting, where box center is
    derived from voting output and the box shape is simply the size of the
    template image.

    To detect potential objects with diversified shapes and scales, we provide
    multiple templates as input.  To further improve the performance and
    robustness, you are also REQUIRED to implement a multi-scale strategy
    either:
       (a) Implement multi-scale interest points and feature descriptors OR
       (b) Repeat a single-scale detection procedure over multiple image scales
           by resizing images.

    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices on multi-scale implementaion and samples of
    detection results (please refer to display_bbox() function in visualize.py).

    Arguments:
        template_images - a list of gray scale images.  Each image is in the
                          form of a 2d numpy array which is cropped to tightly
                          cover the object.

        template_masks  - a list of binary masks having the same shape as the
                          template_image.  Each mask is in the form of 2d numpy
                          array specyfing the foreground mask of object in the
                          corresponding template image.

        test_img        - a gray scale test image in the form of 2d numpy array
                          containing the object category of interest.

    Returns:
         bbox           - a numpy array of shape (4,) specifying the detected
                          bounding box in the format of
                             (x_min, y_min, x_max, y_max)

"""
def object_detection(template_images, template_masks, test_img):
   ##########################################################################
   N = 100
   xs1, ys1, scores1 = find_interest_points(test_img, N, 1.0)
   feats1 = extract_features(test_img, xs1, ys1, 1.0)
   x_min = 0
   x_max = 0
   y_min = 0
   y_max = 0
   most_xvotes = 0
   most_yvotes = 0
   for img in range(len(template_images)):
    for i in range(1,3):
      x, y = np.shape(template_images[img])
      mx, my = np.shape(template_masks[img])
      new_size = np.resize(template_images[img],((x*i),(y*i)))
      new_mask = np.resize(template_masks[img],((mx*i),(my*i)))

      xs0, ys0, scores0 = find_interest_points(new_size, N, 1.0, mask = new_mask)
      if i == 1:
        x_min = min(xs0) 
        x_max = max(xs0) 
        y_min = min(ys0)
        y_max = max(ys0)
      feats0 = extract_features(new_size, xs0, ys0, 1.0)
      matches, match_scores = match_features(feats0, feats1, scores0, scores1)
      temp_tx, temp_ty, votes = hough_votes(xs0, ys0, xs1, ys1, matches, match_scores)
      if votes[0] > most_xvotes:
       tx = temp_tx
       most_xvotes = votes[0]
      if votes[1] > most_yvotes:
       ty = temp_ty
       most_yvotes = votes[1]
   print(x_min, x_max, y_min, y_max)
   print(tx, ty)
   bbox = np.array([int(x_min + tx), int(y_min + ty), int(x_max + tx), int(y_max + ty)])
   ##########################################################################
   return bbox
