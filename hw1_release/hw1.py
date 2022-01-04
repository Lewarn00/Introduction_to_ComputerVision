import numpy as np

"""
   Mirror an image about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of the specified width created by mirroring the interior
"""
def mirror_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   # mirror top/bottom
   top    = image[:wx:,:]
   bottom = image[(sx-wx):,:]
   img = np.concatenate( \
      (top[::-1,:], image, bottom[::-1,:]), \
      axis=0 \
   )
   # mirror left/right
   left  = img[:,:wy]
   right = img[:,(sy-wy):]
   img = np.concatenate( \
      (left[:,::-1], img, right[:,::-1]), \
      axis=1 \
   )
   return img

"""
   Pad an image with zeros about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of zeros
"""
def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img

"""
   Remove the border of an image.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx - 2*wx, sy - 2*wy), extracted by
              removing a border of the specified width from the sides of the
              input image
"""
def trim_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
   return img

"""
   Return an approximation of a 1-dimensional Gaussian filter.

   The returned filter approximates:

   g(x) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2) / (2 * sigma^2) )

   for x in the range [-3*sigma, 3*sigma]
"""
def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

"""
   CONVOLUTION IMPLEMENTATION (10 Points)

   Convolve a 2D image with a 2D filter.

   Requirements:

   (1) Return a result the same size as the input image.

   (2) You may assume the filter has odd dimensions.

   (3) The result at location (x,y) in the output should correspond to
       aligning the center of the filter over location (x,y) in the input
       image.

   (4) When computing a product at locations where the filter extends beyond
       the defined image, treat missing terms as zero.  (Equivalently stated,
       treat the image as being padded with zeros around its border).

   You must write the code for the nested loops of the convolutions yourself,
   using only basic loop constructs, array indexing, multiplication, and
   addition operators.  You may not call any Python library routines that
   implement convolution.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions
      mode   - 'zero': preprocess using pad_border or 'mirror': preprocess using mirror_border.

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
def conv_2d(image, filt, mode='zero'):
  assert image.ndim == 2, 'image should be grayscale'
  filt = np.atleast_2d(filt)

  r = filt.shape[0]
  c = filt.shape[1]
  if mode == 'zero':
    image_with_padding = pad_border(image, r//2, c//2)
  elif mode == 'mirror':
    image_with_padding = mirror_border(image, r//2, c//2)

  output = np.zeros_like(image)
  for x in range(image.shape[1]):
    for y in range(image.shape[0]):
      output[y, x]=(filt * image_with_padding[y: y+r, x: x+c]).sum()

  return output

"""
   GAUSSIAN DENOISING (5 Points)

   Denoise an image by convolving it with a 2D Gaussian filter.

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   You may approximate the G(x,y) filter by computing it on a
   discrete grid for both x and y in the range [-3*sigma, 3*sigma].

   See the gaussian_1d function for reference.

   Note:
   (1) Remember that the Gaussian is a separable filter.
   (2) Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border.

   Arguments:
      image - a 2D numpy array
      sigma - standard deviation of the Gaussian

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input
"""
def gaussian_2D(sigma = 1.0):
  #width = np.ceil(3.0 * sigma)
  #x = np.arange(-width, width + 1)
  #y = np.arange(-width, width + 1)
  #g = np.exp(-((x * x) + (y * y)) / (2 * sigma * sigma))
  #g = g / np.sum(g)     
  #g = np.atleast_2d(g)
  tranposed = gaussian_1d(sigma).T
  normal = gaussian_1d(sigma)
  g = tranposed @ normal

  return g

def denoise_gaussian(image, sigma):
  filt = gaussian_2D(sigma)
  img = conv_2d(image, filt, 'mirror')

  return img

"""
    BILATERAL DENOISING (5 Points)
    Denoise an image by applying a bilateral filter
    Note:
        Performs standard bilateral filtering of an input image.
        Reference link: https://en.wikipedia.org/wiki/Bilateral_filter

        Basically, the idea is adding an additional edge term to Guassian filter
        described above.

        The weighted average pixels:

        BF[I]_p = 1/(W_p)sum_{q in S}G_s(||p-q||)G_r(|I_p-I_q|)I_q

        In which, 1/(W_p) is normalize factor, G_s(||p-q||) is spatial Guassian
        term, G_r(|I_p-I_q|) is range Guassian term.

        We only require you to implement the grayscale version, which means I_p
        and I_q is image intensity.

    Arguments:
        image       - input image
        sigma_s     - spatial param (pixels), spatial extent of the kernel,
                       size of the considered neighborhood.
        sigma_r     - range param (not normalized, a propotion of 0-255),
                       denotes minimum amplitude of an edge
    Returns:
        img   - denoised image, a 2D numpy array of the same shape as the input
"""

def denoise_bilateral(image, sigma_s=1, sigma_r=25.5):
    assert image.ndim == 2, 'image should be grayscale'

    img = np.zeros_like(image)
    m = image.shape[0]
    n = image.shape[1]
    gaussian = gaussian_2D(sigma_s)
    m2 = gaussian.shape[0] // 2
    n2 = gaussian.shape[1] // 2
    slicingM = m2+m2+1
    slicingN = n2+n2+1
    image2 = mirror_border(image, m, n)
    for col in range(n):
      for row in range(m):
        sliced = image2[row:row+slicingM, col:col+slicingN]
        filt = np.zeros_like(gaussian)
        keep_sums, n = 0, 0
        for j in range(m2):
          for i in range(n2):
            filt[i,j] = gaussian[i,j] * np.exp(-(image2[row+m2,col+n2] - image2[row+i,col+j])**2)/(2*sigma_r**2)
            keep_sums += sliced[i,j] * filt[i,j]
            n += filt[i,j]
        img[row,col] = keep_sums / n

    return img

"""
   SMOOTHING AND DOWNSAMPLING (5 Points)

   Smooth an image by applying a gaussian filter, followed by downsampling with a factor k.

   Note:
      Image downsampling is generally implemented as two-step process:

        (1) Smooth images with low pass filter, e.g, gaussian filter, to remove
            the high frequency signal that would otherwise causes aliasing in
            the downsampled outcome.

        (2) Downsample smoothed images by keeping every kth samples.

      Make an appropriate choice of sigma to avoid insufficient or over smoothing.

      In principle, the sigma in gaussian filter should respect the cut-off frequency
      1 / (2 * k) with k being the downsample factor and the cut-off frequency of
      gaussian filter is 1 / (2 * pi * sigma).


   Arguments:
     image - a 2D numpy array
     downsample_factor - an integer specifying downsample rate

   Returns:
     result - downsampled image, a 2D numpy array with spatial dimension reduced
"""
def smooth_and_downsample(image, downsample_factor = 2):
    k = 1 / (2 * downsample_factor)
    smooth = denoise_gaussian(image, k)
    output = []

    for x in range(image.shape[0]):
      if x % downsample_factor == 0:
        output.append([])
        for y in range(image.shape[1]):
          if y % downsample_factor == 0:
            if x > 0:
              output[int(x / downsample_factor)].append(smooth[x, y])
            else:
              output[x].append(smooth[x, y])

    return output

"""
   SOBEL GRADIENT OPERATOR (5 Points)
   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.
   The Sobel operator estimates gradients dx(horizontal), dy(vertical), of
   an image I as:

         [ 1  0  -1 ]
   dx =  [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

         [  1  2  1 ]
   dy =  [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

   where (*) denotes convolution.
   Note:
      (1) Your implementation should be as efficient as possible.
      (2) Avoid creating artifacts along the border of the image.
   Arguments:
      image - a 2D numpy array
   Returns:
      dx    - gradient in x-direction at each point
              (a 2D numpy array, the same shape as the input image)
      dy    - gradient in y-direction at each point
              (a 2D numpy array, the same shape as the input image)
"""
def sobel_gradients(image):
  gx = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])
  gy = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])
  dx = conv_2d(image, gx * -1, mode='mirror')
  dy = conv_2d(image, gy * -1, mode='mirror')

  return dx, dy

"""
   NONMAXIMUM SUPPRESSION (10 Points)

   Nonmaximum suppression.

   Given an estimate of edge strength (mag) and direction (theta) at each
   pixel, suppress edge responses that are not a local maximum along the
   direction perpendicular to the edge.

   Equivalently stated, the input edge magnitude (mag) represents an edge map
   that is thick (strong response in the vicinity of an edge).  We want a
   thinned edge map as output, in which edges are only 1 pixel wide.  This is
   accomplished by suppressing (setting to 0) the strength of any pixel that
   is not a local maximum.

   Note that the local maximum check for location (x,y) should be performed
   not in a patch surrounding (x,y), but along a line through (x,y)
   perpendicular to the direction of the edge at (x,y).

   A simple, and sufficient strategy is to check if:
      ((mag[x,y] > mag[x + ox, y + oy]) and (mag[x,y] >= mag[x - ox, y - oy]))
   or
      ((mag[x,y] >= mag[x + ox, y + oy]) and (mag[x,y] > mag[x - ox, y - oy]))
   where:
      (ox, oy) is an offset vector to the neighboring pixel in the direction
      perpendicular to edge direction at location (x, y)

   Arguments:
      mag    - a 2D numpy array, containing edge strength (magnitude)
      theta  - a 2D numpy array, containing edge direction in [0, 2*pi)

   Returns:
      nonmax - a 2D numpy array, containing edge strength (magnitude), where
               pixels that are not a local maximum of strength along an
               edge have been suppressed (assigned a strength of zero)
"""
def nonmax_suppress(mag, theta):
  nonmax = np.zeros((mag.shape[0],mag.shape[1]))
  theta[theta < 0] += np.pi
  for i in range(1,mag.shape[0]-1):
    for j in range(1,mag.shape[1]-1):
      ox = 1
      oy = 1
      neighbour1 = 255
      neighbour2 = 255
      
      if (0 <= theta[i,j] < np.pi/8) or (7 * np.pi/8 <= theta[i,j] <= np.pi):
        neighbour1 = mag[i, j+oy]
        neighbour2 = mag[i, j-oy]
      elif (np.pi/8 <= theta[i,j] < 3 * np.pi/8):
        neighbour1 = mag[i+ox, j-oy]
        neighbour2 = mag[i-ox, j+oy]
      elif (3 * np.pi/8 <= theta[i,j] < 5 * np.pi/8):
        neighbour1 = mag[i+ox, j]
        neighbour2 = mag[i-ox, j]
      elif (5 * np.pi/8 <= theta[i,j] < 7 * np.pi/8):
        neighbour1 = mag[i-ox, j-oy]
        neighbour2 = mag[i+ox, j+oy]
      
      if mag[i,j] >= neighbour1 and mag[i,j] > neighbour2:
        nonmax[i,j] = mag[i,j]
      else:
        nonmax[i,j] = 0 

  return nonmax


"""
   HYSTERESIS EDGE LINKING (10 Points)

   Hysteresis edge linking.

   Given an edge magnitude map (mag) which is thinned by nonmaximum suppression,
   first compute the low threshold and high threshold so that any pixel below
   low threshold will be thrown away, and any pixel above high threshold is
   a strong edge and will be preserved in the final edge map.  The pixels that
   fall in-between are considered as weak edges.  We then add weak edges to
   true edges if they connect to a strong edge along the gradient direction.

   Since the thresholds are highly dependent on the statistics of the edge
   magnitude distribution, we recommend to consider features like maximum edge
   magnitude or the edge magnitude histogram in order to compute the high
   threshold.  Heuristically, once the high threshod is fixed, you may set the
   low threshold to be propotional to the high threshold.

   Note that the thresholds critically determine the quality of the final edges.
   You need to carefully tuned your threshold strategy to get decent
   performance on real images.

   For the edge linking, the weak edges caused by true edges will connect up
   with a neighbouring strong edge pixel.  To track theses edges, we
   investigate the 8 neighbours of strong edges.  Once we find the weak edges,
   located along strong edges' gradient direction, we will mark them as strong
   edges.  You can adopt the same gradient checking strategy used in nonmaximum
   suppression.  This process repeats util we check all strong edges.

   In practice, we use a queue to implement edge linking.  In python, we could
   use a list and its fuction .append or .pop to enqueue or dequeue.

   Arguments:
     nonmax - a 2D numpy array, containing edge strength (magnitude) which is thined by nonmaximum suppression
     theta  - a 2D numpy array, containing edeg direction in [0, 2*pi)

   Returns:
     edge   - a 2D numpy array, containing edges map where the edge pixel is 1 and 0 otherwise.
"""

def hysteresis_edge_linking(nonmax, theta):
  ##########################################################################
  find_max = nonmax.max()
  high_threshold = find_max * 0.20
  low_threshold = nonmax * 0.05  
  strong = 255
  weak = 2

  edge = np.zeros((nonmax.shape[0],nonmax.shape[1]))
  strong1, strong2 = np.where(nonmax >= high_threshold)
  weak1, weak2 = np.where((nonmax <= high_threshold) & (nonmax >= low_threshold))
  edge[strong1, strong2] = strong
  edge[weak1, weak2] = weak

  theta[theta < 0] += np.pi
  for i in range(1, nonmax.shape[0]-1):
    for j in range(1, nonmax.shape[1]-1):
      if (edge[i,j] == strong):
        if (edge[i+1, j] == weak) and (3 * np.pi/8 <= theta[i,j] < 5 * np.pi/8):
          edge[i+1, j] = strong
        elif (edge[i-1, j] == weak) and (3 * np.pi/8 <= theta[i,j] < 5 * np.pi/8):
          edge[i-1, j] = strong
        elif (edge[i+1, j-1] == weak) and (np.pi/8 <= theta[i,j] < 3 * np.pi/8):
          edge[i+1, j-1] = strong 
        elif (edge[i-1, j+1] == weak) and (np.pi/8 <= theta[i,j] < 3 * np.pi/8):
          edge[i-1, j+1] = strong 

        elif (edge[i, j-1] == weak) and ((0 <= theta[i,j] < np.pi/8) or (7 * np.pi/8 <= theta[i,j] <= np.pi)):
          edge[i, j-1] = strong 
        elif (edge[i, j+1] == weak) and ((0 <= theta[i,j] < np.pi/8) or (7 * np.pi/8 <= theta[i,j] <= np.pi)):
          edge[i, j+1] = strong 

        elif (edge[i-1, j-1] == weak) and (5 * np.pi/8 <= theta[i,j] < 7 * np.pi/8):
          edge[i-1, j-1] = strong  
        elif (edge[i+1, j+1] == weak) and (5 * np.pi/8 <= theta[i,j] < 7 * np.pi/8):
          edge[i+1, j+1] = strong

        else:
          edge[i, j] = 0

  for i in range(1, nonmax.shape[0]-1):
    for j in range(1, nonmax.shape[1]-1):
      if (edge[i,j] == weak):
        edge[i, j] = 0

  return edge 

"""
   CANNY EDGE DETECTOR (5 Points)

   Canny edge detector.

   Given an input image:

   (1) Compute gradients in x- and y-directions at every location using the
       Sobel operator.  See sobel_gradients() above.

   (2) Estimate edge strength (gradient magnitude) and direction.

   (3) Perform nonmaximum suppression of the edge strength map, thinning it
       in the direction perpendicular to that of a local edge.
       See nonmax_suppress() above.

   (4) Compute the high threshold and low threshold of edge strength map
       to classify the pixels as strong edges, weak edges and non edges.
       Then link weak edges to strong edges

   Return the original edge strength estimate (max), the edge
   strength map after nonmaximum suppression (nonmax) and the edge map
   after edge linking (edge)

   Arguments:
      image    - a 2D numpy array

   Returns:
      mag      - a 2D numpy array, same shape as input, edge strength at each pixel
      nonmax   - a 2D numpy array, same shape as input, edge strength after nonmaximum suppression
      edge     - a 2D numpy array, same shape as input, edges map where edge pixel is 1 and 0 otherwise.
"""
def canny(image):
  dx, dy = sobel_gradients(image)
  mag = np.sqrt(dx * dx + dy * dy)
  theta = np.arctan2(dx, dy)
  nonmax = nonmax_suppress(mag, theta)
  edge = hysteresis_edge_linking(nonmax, theta)

  return mag, nonmax, edge
