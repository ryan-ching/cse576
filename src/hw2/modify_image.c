#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
  // TODO
  /***********************************************************************
      This function performs nearest-neighbor interpolation on image "im"
      given a floating column value "x", row value "y" and integer channel "c",
      and returns the interpolated value.
    ************************************************************************/
  return get_pixel(im, round(x), round(y), c);
}

image nn_resize(image im, int w, int h)
{
  // TODO Fill in (also fix the return line)
  /***********************************************************************
      This function uses nearest-neighbor interpolation on image "im" to a new
      image of size "w x h"
    ************************************************************************/
  // ax + b = y, y = coordinate of old image
  // a = (width of old image) / (width of resized)
  // b = ax - y => @(1  ,1   ) b = a - 1 = > 0.5 offset (from class -3/14 = (-3/14 = 4/7 - 1) -> (a - 1)(0.5)
  image resized = make_image(w, h, im.c);
  float a1 = im.w / (float)w;
  float a2 = im.h / (float)h;
  float b1 = 0.5 * (a1 - 1);
  float b2 = 0.5 * (a2 - 1);
  int currW, currH, currC;
  for (currW = 0; currW < w; currW++)
  {
    for (currH = 0; currH < h; currH++)
    {
      for (currC = 0; currC < im.c; currC++)
      {
        set_pixel(resized, currW, currH, currC, // rint because 0.5 should be 0
                  nn_interpolate(im, currW * a1 + b1, currH * a2 + b2, currC));
      }
    }
  }
  return resized;
}

float bilinear_interpolate(image im, float x, float y, int c)
{

  // TODO
  /***********************************************************************
      This function performs bilinear interpolation on image "im" given
      a floating column value "x", row value "y" and integer channel "c".
      It interpolates and returns the interpolated value.
    ************************************************************************/

  float v1 = get_pixel(im, floorf(x), floorf(y), c); //  Top Left
  float v2 = get_pixel(im, ceilf(x), floorf(y), c);  // Top Right
  float v3 = get_pixel(im, floorf(x), ceilf(y), c);  // Bottom Left
  float v4 = get_pixel(im, ceilf(x), ceilf(y), c);   // Bottom Right

  float d1 = x - floorf(x);
  float d2 = ceilf(x) - x;
  float d3 = y - floorf(y);
  float d4 = ceilf(y) - y;

  float a1 = d2 * d4; // Bottom Right
  float a2 = d1 * d4; // Bottom Left
  float a3 = d2 * d3; // Top Right
  float a4 = d1 * d3; // Top Left

  return v1 * a1 + v2 * a2 + v3 * a3 + v4 * a4;
}

image bilinear_resize(image im, int w, int h)
{
  // TODO
  /***********************************************************************
      This function uses bilinear interpolation on image "im" to a new image
      of size "w x h". Algorithm is same as nearest-neighbor interpolation.
    ************************************************************************/
  image resized = make_image(w, h, im.c);
  float a1 = im.w / (float)w;
  float a2 = im.h / (float)h;
  float b1 = 0.5 * (a1 - 1);
  float b2 = 0.5 * (a2 - 1);
  int currW, currH, currC;
  for (currW = 0; currW < w; currW++)
  {
    for (currH = 0; currH < h; currH++)
    {
      for (currC = 0; currC < im.c; currC++)
      {
        set_pixel(resized, currW, currH, currC,
                  bilinear_interpolate(im, currW * a1 + b1, currH * a2 + b2, currC));
      }
    }
  }
  return resized;
}
/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
  // TODO
  /***********************************************************************
      This function divides each value in image "im" by the sum of all the
      values in the image and modifies the image in place.
    ************************************************************************/

  int x, y, c;
  float sum = 0.0;
  for (x = 0; x < im.w; x++)
  {
    for (y = 0; y < im.h; y++)
    {
      for (c = 0; c < im.c; c++)
      {
        sum += get_pixel(im, x, y, c); // Sum of all values in the image
      }
    }
  }
  for (x = 0; x < im.w; x++)
  {
    for (y = 0; y < im.h; y++)
    {
      for (c = 0; c < im.c; c++)
      {
        set_pixel(im, x, y, c, get_pixel(im, x, y, c) / sum);
      }
    }
  }
}

image make_box_filter(int w)
{
  // TODO
  /***********************************************************************
      This function makes a square filter of size "w x w". Make an image of
      width = height = w and number of channels = 1, with all entries equal
      to 1. Then use "l1_normalize" to normalize your filter.
    ************************************************************************/
  image ones = make_image(w, w, 1);
  int x, y;
  for (x = 0; x < ones.w; x++)
  {
    for (y = 0; y < ones.h; y++)
    {
      set_pixel(ones, x, y, 0, 1);
    }
  }
  l1_normalize(ones);
  return ones;
}

image convolve_image(image im, image filter, int preserve)
{
  // TODO
  /***********************************************************************
      This function convolves the image "im" with the "filter". The value
      of preserve is 1 if the number of input image channels need to be 
      preserved. Check the detailed algorithm given in the README.  
    ************************************************************************/
  assert(filter.c == im.c || filter.c == 1);
  assert(preserve == 0 || preserve == 1);
  int x, y, c, xf, yf, xc, yc;
  float cv, sum, nosum;                                                                         // cv = Convolved Value (No sum across channels)
  image convolved = (preserve == 1) ? make_image(im.w, im.h, im.c) : make_image(im.w, im.h, 1); // 1 channel (no preserve), multi-channel (preserve)
  for (x = 0; x < im.w; x++)
  {
    for (y = 0; y < im.h; y++)
    {
      sum = 0.0;
      for (c = 0; c < im.c; c++)
      {
        cv = 0.0;
        for (xf = 0; xf < filter.w; xf++) // Go through filter w/h dimensions and multiply image by filter
        {
          for (yf = 0; yf < filter.h; yf++)
          {
            xc = x + xf - (int)(filter.w / 2); // Center around pixel being convolved around
            yc = y + yf - (int)(filter.h / 2);
            cv += (filter.c == im.c) ? get_pixel(im, xc, yc, c) * get_pixel(filter, xf, yf, c)
                                     : get_pixel(im, xc, yc, c) * get_pixel(filter, xf, yf, 0);
          }
        }
        sum += cv;         // CV is now the convolved value after looping through filter dim
        if (preserve == 1) // Apply filter to each channel (Result is multi-channel image)
        {
          set_pixel(convolved, x, y, c, cv);
        }
      }
      if (preserve == 0)
      { // 1 Channel Image, sum between channels
        set_pixel(convolved, x, y, 0, sum);
      }
    }
  }
  return convolved;
}

image make_highpass_filter()
{
  // TODO
  /***********************************************************************
      Create a 3x3 filter with highpass filter values using image.data[]
    ************************************************************************/
  //  0 -1  0
  // -1  4 -1
  //  0 -1  0
  image filter = make_image(3, 3, 1);
  set_pixel(filter, 0, 0, 0, 0.0);
  set_pixel(filter, 1, 0, 0, -1.0);
  set_pixel(filter, 2, 0, 0, 0.0);
  set_pixel(filter, 0, 1, 0, -1.0);
  set_pixel(filter, 1, 1, 0, 4.0);
  set_pixel(filter, 2, 1, 0, -1.0);
  set_pixel(filter, 0, 2, 0, 0.0);
  set_pixel(filter, 1, 2, 0, -1.0);
  set_pixel(filter, 2, 2, 0, 0.0);
  return filter;
}

image make_sharpen_filter()
{
  // TODO
  /***********************************************************************
      Create a 3x3 filter with sharpen filter values using image.data[]
    ************************************************************************/
  //  0 -1  0
  // -1  5 -1
  //  0 -1  0
  image filter = make_image(3, 3, 1);
  set_pixel(filter, 0, 0, 0, 0.0);
  set_pixel(filter, 1, 0, 0, -1.0);
  set_pixel(filter, 2, 0, 0, 0.0);
  set_pixel(filter, 0, 1, 0, -1.0);
  set_pixel(filter, 1, 1, 0, 5.0);
  set_pixel(filter, 2, 1, 0, -1.0);
  set_pixel(filter, 0, 2, 0, 0.0);
  set_pixel(filter, 1, 2, 0, -1.0);
  set_pixel(filter, 2, 2, 0, 0.0);
  return filter;
}

image make_emboss_filter()
{
  // TODO
  /***********************************************************************
      Create a 3x3 filter with emboss filter values using image.data[]
    ************************************************************************/
  // -2 -1  0
  // -1  1  1
  //  0  1  2
  image filter = make_image(3, 3, 1);
  set_pixel(filter, 0, 0, 0, -2.0);
  set_pixel(filter, 1, 0, 0, -1.0);
  set_pixel(filter, 2, 0, 0, 0.0);
  set_pixel(filter, 0, 1, 0, -1.0);
  set_pixel(filter, 1, 1, 0, 1.0);
  set_pixel(filter, 2, 1, 0, 1.0);
  set_pixel(filter, 0, 2, 0, 0.0);
  set_pixel(filter, 1, 2, 0, 1.0);
  set_pixel(filter, 2, 2, 0, 2.0);
  return filter;
}

// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: Any filter in which you want to preserve the color, you should use preserve, as it will keep the RBG channels in the filtered image.
// Preserve: Sharpen - Sharpens image while preserving color
//           Emboss - Styling applied to all three bands
// No Preserve: Highpass - We are just finding edges, no need to preserve color

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: Highpass - Threshold to find strong vs weak edges

image make_gaussian_filter(float sigma)
{
  // TODO
  /***********************************************************************
      sigma: a float number for the Gaussian.
      Create a Gaussian filter with the given sigma. Note that the kernel size 
      is the next highest odd integer from 6 x sigma. Return the Gaussian filter.
    ************************************************************************/

  int kernel_dim = ((int)roundf(sigma * 6) % 2 == 0) ? (int)roundf(sigma * 6) + 1 : (int)roundf(sigma * 6);
  image kernel = make_image(kernel_dim, kernel_dim, 1);
  int x, y, c, xo, yo;
  for (x = 0; x < kernel_dim; x++)
  {
    for (y = 0; y < kernel_dim; y++)
    {

      xo = x - (int)(0.5 * kernel_dim); // 0.5 Offset
      yo = y - (int)(0.5 * kernel_dim);
      float val = (1.0 / (TWOPI * pow(sigma, 2)) * exp(-(pow(xo, 2) + pow(yo, 2)) / (2 * pow(sigma, 2))));
      set_pixel(kernel, x, y, 0, val);
    }
  }
  l1_normalize(kernel);
  return kernel;
}

image add_image(image a, image b)
{
  // TODO
  /***********************************************************************
      The input images a and image b have the same height, width, and channels.
      Sum the given two images and return the result, which should also have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
  assert(a.w == b.w && a.h == b.h && a.c == b.c);
  image res = make_image(a.w, a.h, a.c);
  int x, y, c;
  float val;
  for (x = 0; x < a.w; x++)
  {
    for (y = 0; y < b.w; y++)
    {
      for (c = 0; c < a.c; c++)
      {
        val = get_pixel(a, x, y, c) + get_pixel(b, x, y, c);
        set_pixel(res, x, y, c, val);
      }
    }
  }
  return res;
}

image sub_image(image a, image b)
{
  // TODO
  /***********************************************************************
      The input image a and image b have the same height, width, and channels.
      Subtract the given two images and return the result, which should have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
  assert(a.w == b.w && a.h == b.h && a.c == b.c);
  image res = make_image(a.w, a.h, a.c);
  int x, y, c;
  float val;
  for (x = 0; x < a.w; x++)
  {
    for (y = 0; y < b.w; y++)
    {
      for (c = 0; c < a.c; c++)
      {
        val = get_pixel(a, x, y, c) - get_pixel(b, x, y, c);
        set_pixel(res, x, y, c, val);
      }
    }
  }
  return res;
}

image make_gx_filter()
{
  // TODO
  /***********************************************************************
      Create a 3x3 Sobel Gx filter and return it
    ************************************************************************/
  // -1  0  1
  // -2  0  2
  // -1  0  1
  image filter = make_image(3, 3, 1);
  set_pixel(filter, 0, 0, 0, -1.0);
  set_pixel(filter, 1, 0, 0, 0.0);
  set_pixel(filter, 2, 0, 0, 1.0);
  set_pixel(filter, 0, 1, 0, -2.0);
  set_pixel(filter, 1, 1, 0, 0.0);
  set_pixel(filter, 2, 1, 0, 2.0);
  set_pixel(filter, 0, 2, 0, -1.0);
  set_pixel(filter, 1, 2, 0, 0.0);
  set_pixel(filter, 2, 2, 0, 1.0);
  return filter;
}

image make_gy_filter()
{
  // TODO
  /***********************************************************************
      Create a 3x3 Sobel Gy filter and return it
    ************************************************************************/
  // -1 -2 -1
  //  0  0  0
  //  1  2  1
  image filter = make_image(3, 3, 1);
  set_pixel(filter, 0, 0, 0, -1.0);
  set_pixel(filter, 1, 0, 0, -2.0);
  set_pixel(filter, 2, 0, 0, -1.0);
  set_pixel(filter, 0, 1, 0, 0.0);
  set_pixel(filter, 1, 1, 0, 0.0);
  set_pixel(filter, 2, 1, 0, 0.0);
  set_pixel(filter, 0, 2, 0, 1.0);
  set_pixel(filter, 1, 2, 0, 2.0);
  set_pixel(filter, 2, 2, 0, 1.0);
  return filter;
}

void feature_normalize(image im)
{
  // TODO
  /***********************************************************************
      Calculate minimum and maximum pixel values. Normalize the image by
      subtracting the minimum and dividing by the max-min difference.
    ************************************************************************/
  float min = get_pixel(im, 0, 0, 0);
  float max = get_pixel(im, 0, 0, 0);
  float curr, diff;
  int x, y, c;
  for (x = 0; x < im.w; x++)
  {
    for (y = 0; y < im.w; y++)
    {
      for (c = 0; c < im.c; c++)
      {
        curr = get_pixel(im, x, y, c);
        min = (curr < min) ? curr : min;
        max = (curr > max) ? curr : max;
      }
    }
  }
  for (x = 0; x < im.w; x++)
  {
    for (y = 0; y < im.w; y++)
    {
      for (c = 0; c < im.c; c++)
      {
        curr = get_pixel(im, x, y, c);
        set_pixel(im, x, y, c, (curr - min) / (max - min));
      }
    }
  }
}

image *sobel_image(image im)
{
  // TODO
  /***********************************************************************
      Apply Sobel filter to the input image "im", get the magnitude as sobelimg[0]
      and gradient as sobelimg[1], and return the result.
    ************************************************************************/
  image *sobelimg = calloc(2, sizeof(image));
  image gx = convolve_image(im, make_gx_filter(), 0); // Gradient values of input (only 1 channel)
  image gy = convolve_image(im, make_gy_filter(), 0);
  image mag = make_image(im.w, im.h, 1);
  image dir = make_image(im.w, im.h, 1);

  int x, y, c;
  float gx_curr, gy_curr;
  for (x = 0; x < im.w; x++)
  {
    for (y = 0; y < im.h; y++)
    {
      gx_curr = get_pixel(gx, x, y, 0); // 1 channel result (no preserve)
      gy_curr = get_pixel(gy, x, y, 0);
      set_pixel(mag, x, y, 0, sqrtf(pow(gx_curr, 2) + pow(gy_curr, 2))); // G = sqrt(gx^2 + gy^2) (Magnitude)
      set_pixel(dir, x, y, 0, atan2f(gy_curr, gx_curr));                 // theta = atan2(gy,gx) (Direction)
    }
  }
  *(sobelimg) = mag;
  *(sobelimg + 1) = dir;
  return sobelimg;
}

image colorize_sobel(image im)
{
  // TODO
  /***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
  ************************************************************************/
  image *sobel = sobel_image(im);
  image saturation = *sobel;
  image hue = *(sobel + 1);
  image hsv_im = make_image(im.w, im.h, im.c);
  int x, y, c;
  for (x = 0; x < im.w; x++)
  {
    for (y = 0; y < im.w; y++)
    {
      //v = three_way_max(get_pixel(im, x, y, 0), get_pixel(im, x, y, 1), get_pixel(im, x, y, 2));
      set_pixel(hsv_im, x, y, 0, get_pixel(hue, x, y, 0));        // Hue
      set_pixel(hsv_im, x, y, 1, get_pixel(saturation, x, y, 0)); // Saturation
      set_pixel(hsv_im, x, y, 2, get_pixel(saturation, x, y, 0)); // Value (equal to magnitude)
    }
  }
  hsv_to_rgb(hsv_im);
  return convolve_image(hsv_im, make_gaussian_filter(4.0), 1);
  ;
}

// EXTRA CREDIT: Median filter

/*
image apply_median_filter(image im, int kernel_size)
{
  return make_image(1,1,1);
}
*/

// SUPER EXTRA CREDIT: Bilateral filter

/*
image apply_bilateral_filter(image im, float sigma1, float sigma2)
{
  return make_image(1,1,1);
}
*/
