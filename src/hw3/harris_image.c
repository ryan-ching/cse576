#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i % im.w;
    d.p.y = i / im.w;
    d.data = calloc(w * w * im.c, sizeof(float));
    d.n = w * w * im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for (c = 0; c < im.c; ++c)
    {
        float cval = im.data[c * im.w * im.h + i];
        for (dx = -w / 2; dx < (w + 1) / 2; ++dx)
        {
            for (dy = -w / 2; dy < (w + 1) / 2; ++dy)
            {
                float val = get_pixel(im, i % im.w + dx, i / im.w + dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for (i = -9; i < 10; ++i)
    {
        set_pixel(im, x + i, y, 0, 1);
        set_pixel(im, x, y + i, 0, 1);
        set_pixel(im, x + i, y, 1, 0);
        set_pixel(im, x, y + i, 1, 0);
        set_pixel(im, x + i, y, 2, 1);
        set_pixel(im, x, y + i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    // TODO: make separable 1d Gaussian.
    int kernel_dim = ((int)ceilf(sigma * 6) % 2 == 0) ? (int)ceilf(sigma * 6) + 1 : (int)ceilf(sigma * 6);
    image gauss = make_image(kernel_dim, 1, 1); // Creating N x 1 Matrix
    int x, xo;
    for (x = 0; x < gauss.w; x++)
    {
        xo = (int)(0.5 * kernel_dim) - x; // Offset
        float val = (1.0 / (sqrtf(TWOPI) * sigma) * exp(-1 * pow(xo, 2) / (2 * pow(sigma, 2))));
        set_pixel(gauss, x, 0, 0, val);
    }
    return gauss;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    // TODO: use two convolutions with 1d gaussian filter.
    image g_1_n = make_1d_gaussian(sigma);
    image g_n_1 = make_image(1, g_1_n.w, 1);
    int y;
    for (y = 0; y < g_n_1.h; y++)
    {
        set_pixel(g_n_1, 0, y, 0, get_pixel(g_1_n, y, 0, 0));
    }
    image temp = convolve_image(im, g_1_n, 1);
    image smoothed = convolve_image(temp, g_n_1, 1);
    return smoothed;
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    image S = make_image(im.w, im.h, 3);
    // TODO: calculate structure matrix for im.
    // Calculate Image derivatives Ix and Iy
    // Calculate measures Ix^2, Iy^2 and Ix * Iy
    // Calculate structure matrix S as weighted sum of nearby measures
    image ix = convolve_image(im, make_gx_filter(), 0);
    image iy = convolve_image(im, make_gy_filter(), 0);
    image ix_2 = make_image(ix.w, ix.h, 1);
    image iy_2 = make_image(iy.w, iy.h, 1);
    image ix_iy = make_image(im.w, im.h, 1);

    int x, y;
    float ix_p, iy_p; // Current ix, iy pixel with gaussian
    for (x = 0; x < im.w; x++)
    {
        for (y = 0; y < im.h; y++)
        {
            ix_p = get_pixel(ix, x, y, 0);
            iy_p = get_pixel(iy, x, y, 0);
            set_pixel(ix_2, x, y, 0, powf(ix_p, 2));
            set_pixel(iy_2, x, y, 0, powf(iy_p, 2));
            set_pixel(ix_iy, x, y, 0, ix_p * iy_p);
        }
    }
    for (x = 0; x < im.w; x++)
    {
        for (y = 0; y < im.h; y++)
        {
            set_pixel(S, x, y, 0, get_pixel(ix_2, x, y, 0));
            set_pixel(S, x, y, 1, get_pixel(iy_2, x, y, 0));
            set_pixel(S, x, y, 2, get_pixel(ix_iy, x, y, 0));
        }
    }
    return smooth_image(S, sigma);
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    // TODO: fill in R, "cornerness" for each pixel using the structure matrix.
    // We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
    float det_s, trace_s, res;
    int x, y, c;
    for (x = 0; x < S.w; x++)
    {
        for (y = 0; y < S.h; y++)
        {
            // Det(s) = Ix^2 * Iy^2 - IxIy *IxIy
            det_s = (get_pixel(S, x, y, 0) * get_pixel(S, x, y, 1)) -
                    powf(get_pixel(S, x, y, 2), 2);
            //Tr(S) = Ix^2 + Iy^2
            trace_s = get_pixel(S, x, y, 0) + get_pixel(S, x, y, 1);
            // Res = Det(S) - 0.06 * Tr(S) * Tr(S)
            res = det_s - (0.06 * powf(trace_s, 2));
            set_pixel(R, x, y, 0, res);
        }
    }
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    image r = copy_image(im);
    // TODO: perform NMS on the response map.
    // for every pixel in the image:
    //     for neighbors within w:
    //         if neighbor response greater than pixel response:
    //             set response to be very low (I use -999999 [why not 0??])
    float low = -999999.0;
    int x, y;
    int xn, yn; // Neighborhood bounds
    float res;
    for (x = 0; x < im.w; x++)
    {
        for (y = 0; y < im.h; y++)
        {
            for (xn = x - w; xn < x + w; xn++)
            {
                for (yn = y - w; yn < y + w; yn++)
                {
                    // if neighboring pixel response value > current pixel response value, set response to low
                    res = (get_pixel(im, xn, yn, 0) > get_pixel(im, x, y, 0)) ? low : get_pixel(im, x, y, 0);
                    set_pixel(r, x, y, 0, res);
                }
            }
        }
    }
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);

    //TODO: count number of responses over threshold

    int count = 0;
    int x, y, c;
    for (x = 0; x < Rnms.w; x++)
    {
        for (y = 0; y < Rnms.h; y++)
        {
            for (c = 0; c < Rnms.c; c++)
            {
                if (get_pixel(Rnms, x, y, c) > thresh)
                {
                    count += 1;
                }
            }
        }
    }

    *n = count; // <- set *n equal to number of corners in image.
    descriptor *d = calloc(count, sizeof(descriptor));
    //TODO: fill in array *d with descriptors of corners, use describe_index.
    int index = 0;
    for (x = 0; x < Rnms.w; x++)
    {
        for (y = 0; y < Rnms.h; y++)
        {
            for (c = 0; c < Rnms.c; c++)
            {
                if (get_pixel(Rnms, x, y, c) + thresh)
                {
                    *(d + index) = describe_index(Rnms, x + (y * Rnms.w) + (c * Rnms.w * Rnms.h));
                    index += 1;
                }
            }
        }
    }
    free_image(S);
    free_image(R);
    free_image(Rnms);
    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
