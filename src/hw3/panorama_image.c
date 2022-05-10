#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"

// Comparator for matches
// const void *a, *b: pointers to the matches to compare.
// returns: result of comparison, 0 if same, 1 if a > b, -1 if a < b.
int match_compare(const void *a, const void *b)
{
    match *ra = (match *)a;
    match *rb = (match *)b;
    if (ra->distance < rb->distance)
        return -1;
    else if (ra->distance > rb->distance)
        return 1;
    else
        return 0;
}

// Helper function to create 2d points.
// float x, y: coordinates of point.
// returns: the point.
point make_point(float x, float y)
{
    point p;
    p.x = x;
    p.y = y;
    return p;
}

// Place two images side by side on canvas, for drawing matching pixels.
// image a, b: images to place.
// returns: image with both a and b side-by-side.
image both_images(image a, image b)
{
    image both = make_image(a.w + b.w, a.h > b.h ? a.h : b.h, a.c > b.c ? a.c : b.c);
    int i, j, k;
    for (k = 0; k < a.c; ++k)
    {
        for (j = 0; j < a.h; ++j)
        {
            for (i = 0; i < a.w; ++i)
            {
                set_pixel(both, i, j, k, get_pixel(a, i, j, k));
            }
        }
    }
    for (k = 0; k < b.c; ++k)
    {
        for (j = 0; j < b.h; ++j)
        {
            for (i = 0; i < b.w; ++i)
            {
                set_pixel(both, i + a.w, j, k, get_pixel(b, i, j, k));
            }
        }
    }
    return both;
}

// Draws lines between matching pixels in two images.
// image a, b: two images that have matches.
// match *matches: array of matches between a and b.
// int n: number of matches.
// int inliers: number of inliers at beginning of matches, drawn in green.
// returns: image with matches drawn between a and b on same canvas.
image draw_matches(image a, image b, match *matches, int n, int inliers)
{
    image both = both_images(a, b);
    int i, j;
    for (i = 0; i < n; ++i)
    {
        int bx = matches[i].p.x;
        int ex = matches[i].q.x;
        int by = matches[i].p.y;
        int ey = matches[i].q.y;
        for (j = bx; j < ex + a.w; ++j)
        {
            int r = (float)(j - bx) / (ex + a.w - bx) * (ey - by) + by;
            set_pixel(both, j, r, 0, i < inliers ? 0 : 1);
            set_pixel(both, j, r, 1, i < inliers ? 1 : 0);
            set_pixel(both, j, r, 2, 0);
        }
    }
    return both;
}

// Draw the matches with inliers in green between two images.
// image a, b: two images to match.
// matches *
image draw_inliers(image a, image b, matrix H, match *m, int n, float thresh)
{
    int inliers = model_inliers(H, m, n, thresh);
    image lines = draw_matches(a, b, m, n, inliers);
    return lines;
}

// Find corners, match them, and draw them between two images.
// image a, b: images to match.
// float sigma: gaussian for harris corner detector. Typical: 2
// float thresh: threshold for corner/no corner. Typical: 1-5
// int nms: window to perform nms on. Typical: 3
image find_and_draw_matches(image a, image b, float sigma, float thresh, int nms)
{
    int an = 0;
    int bn = 0;
    int mn = 0;
    descriptor *ad = harris_corner_detector(a, sigma, thresh, nms, &an);
    descriptor *bd = harris_corner_detector(b, sigma, thresh, nms, &bn);
    match *m = match_descriptors(ad, an, bd, bn, &mn);

    mark_corners(a, ad, an);
    mark_corners(b, bd, bn);
    image lines = draw_matches(a, b, m, mn, 0);

    free_descriptors(ad, an);
    free_descriptors(bd, bn);
    free(m);
    return lines;
}

// Calculates L1 distance between to floating point arrays.
// float *a, *b: arrays to compare.
// int n: number of values in each array.
// returns: l1 distance between arrays (sum of absolute differences).
float l1_distance(float *a, float *b, int n)
{
    // TODO: return the correct number.
    float sum;
    int i;
    for (i = 0; i < n; i++)
    {
        sum += fabs(fabs(*(a + i)) - fabs(*(b + i)));
    }
    return sum;
}

// Finds best matches between descriptors of two images.
// descriptor *a, *b: array of descriptors for pixels in two images.
// int an, bn: number of descriptors in arrays a and b.
// int *mn: pointer to number of matches found, to be filled in by function.
// returns: best matches found. each descriptor in a should match with at most
//          one other descriptor in b.
match *match_descriptors(descriptor *a, int an, descriptor *b, int bn, int *mn)
{
    int i, j;

    // We will have at most an matches.
    *mn = an;
    match *m = calloc(an, sizeof(match));
    float distance, min_distance;
    int bind;
    for (j = 0; j < an; ++j)
    {
        // TODO: for every descriptor in a, find best match in b.
        // record ai as the index in *a and bi as the index in *b.
        min_distance = 999999.0; // Initially set to very high
        bind = 0;                // <- find the best match
        for (i = 0; i < bn; ++i)
        {
            distance = l1_distance((a + j)->data, (b + i)->data, sizeof(*(a + j)));
            if (distance < min_distance)
            {
                bind = i;
                min_distance = distance;
            }
        }
        m[j].ai = j;
        m[j].bi = bind; // <- should be index in b.
        m[j].p = a[j].p;
        m[j].q = b[bind].p;
        m[j].distance = min_distance; // <- should be the smallest L1 distance!
    }

    int count = 0;

    // TODO: we want matches to be injective (one-to-one).
    qsort(m, an, sizeof(match), match_compare); // Sort matches based on distance using match_compare and qsort.
    int *seen = calloc(bn, sizeof(int));        // Iniitalize an array of 0s called seen
    // Each point should only be a part of one match.
    // Some points will not be in a match.
    // In practice just bring good matches to front of list, set *mn.
    int mc;
    for (mc = 0; mc < sizeof(*m); ++mc) // Loop over all matches
    {
        if (*(seen + m[mc].bi) != 1) // if b-index of current match != 1 in 'seen
        {
            *(seen + m[mc].bi) = 1; // set corresponding value in 'seen' to 1 (retain match)
        }
        else // Discard matches to the same element in b. Use seen to keep track.
        {
            for (i = mc; i < sizeof(*m) - 1; i++) // Remove and shift elements
            {
                *(m + i) = *(m + i + i);
            }
        }
    }
    *mn = count;
    free(seen);
    return m;
}

// Apply a projective transformation to a point.
// matrix H: homography to project point.
// point p: point to project.
// returns: point projected using the homography.
point project_point(matrix H, point p)
{
    matrix c = make_matrix(3, 1);
    // TODO: project point p with homography H.
    // Remember that homogeneous coordinates are equivalent up to scalar.
    // Have to divide by.... something...
    // Use matrix_mult_matrix
    // normalization
    // Given point p, set matrix c_3_1 = [x-coord, ycoord, 1]
    c.data[0][0] = p.x;
    c.data[1][0] = p.y;
    c.data[2][0] = 1;
    // Compute mmatrix M_3_1 = H_3_3 * C_3_1 with given homography
    matrix m = matrix_mult_matrix(H, c); // 3x1 Output
    // Compute x,y coordinates of a point 'q';
    float x = m.data[0][0] / m.data[2][0]; //  x-coord: M[0] / M[2]
    float y = m.data[1][0] / m.data[2][0]; //  y-coord: M[1] / M[2]
    point q = make_point(x, y);
    return q;
}

// Calculate L2 distance between two points.
// point p, q: points.
// returns: L2 distance between them.
float point_distance(point p, point q)
{
    // TODO: should be a quick one.
    float res = sqrtf(powf(q.x - p.x, 2) + powf(q.y - p.y, 2));
    return res;
}

// Count number of inliers in a set of matches. Should also bring inliers
// to the front of the array.
// matrix H: homography between coordinate systems.
// match *m: matches to compute inlier/outlier.
// int n: number of matches in m.
// float thresh: threshold to be an inlier.
// returns: number of inliers whose projected point falls within thresh of
//          their match in the other image. Should also rearrange matches
//          so that the inliers are first in the array. For drawing.
int model_inliers(matrix H, match *m, int n, float thresh)
{
    int i;
    int count = 0;
    point proj, temp;
    float dist;
    // TODO: count number of matches that are inliers
    // i.e. distance(H*p, q) < thresh
    // Also, sort the matches m so the inliers are the first 'count' elements.
    for (i = n - 1; i >= 0; i--) // Loop over each map from array of matches starting from end
    {
        proj = project_point(H, (m + i)->p);     // Project point p of match using H
        dist = point_distance((m + i)->p, proj); // compute l2 distance between point q of match and the projected point
        if (dist < thresh)                       // if distance < given threshold:
        {
            count++;     // update inlier count
            temp = m->p; //  bring match to front of array (swap)
            m->p = (m + i)->p;
            (m + i)->p = temp;
        }
    }
    return count;
}

// Randomly shuffle matches for RANSAC.
// match *m: matches to shuffle in place.
// int n: number of elements in matches.
void randomize_matches(match *m, int n)
{
    // TODO: implement Fisher-Yates to shuffle the array.
    int i, j;
    match temp;
    for (i = n - 1; i > 0; i--)
    {
        j = rand() % (n + 1);
        temp = *(m + j);
        *(m + j) = *(m + i);
        *(m + i) = temp;
    }
}

// Computes homography between two images given matching pixels.
// match *matches: matching points between images.
// int n: number of matches to use in calculating homography.
// returns: matrix representing homography H that maps image a to image b.
matrix compute_homography(match *matches, int n)
{
    matrix M = make_matrix(n * 2, 8);
    matrix b = make_matrix(n * 2, 1);
    int i, r1, r2;
    for (i = 0; i < n; i++)
    {
        double x = matches[i].p.x;
        double xp = matches[i].q.x;
        double y = matches[i].p.y;
        double yp = matches[i].q.y;
        // TODO: fill in the matrices M and b.
        r1 = 2 * i;
        r2 = (2 * i) + 1;
        b.data[r1][0] = -xp;
        b.data[r2][1] = -yp;

        M.data[r1][0] = x;
        M.data[r1][1] = y;
        M.data[r1][2] = 1;
        M.data[r1][3] = 0;
        M.data[r1][4] = 0;
        M.data[r1][5] = 0;
        M.data[r1][6] = -xp * x;
        M.data[r1][7] = -xp * y;

        M.data[r2][0] = 0;
        M.data[r2][1] = 0;
        M.data[r2][2] = 0;
        M.data[r2][3] = x;
        M.data[r2][4] = y;
        M.data[r2][5] = 1;
        M.data[r2][6] = -yp * x;
        M.data[r2][7] = -yp * y;
    }
    matrix a = solve_system(M, b);
    free_matrix(M);
    free_matrix(b);

    // If a solution can't be found, return empty matrix;
    matrix none = {0};
    if (!a.data)
        return none;

    matrix H = make_matrix(3, 3);
    // TODO: fill in the homography H based on the result in a.
    matrix zeros = make_matrix(n * 2, 0);
    for (i = 0; i < n * 2; i++)
    {
        zeros.data[i][0] = 0;
    }
    H = solve_system(a, zeros);
    free_matrix(a);
    return H;
}

// Perform RANdom SAmple Consensus to calculate homography for noisy matches.
// match *m: set of matches.
// int n: number of matches.
// float thresh: inlier/outlier distance threshold.
// int k: number of iterations to run.
// int cutoff: inlier cutoff to exit early.
// returns: matrix representing most common homography between matches.
matrix RANSAC(match *m, int n, float thresh, int k, int cutoff)
{
    int e, i;
    int best = 0;
    matrix Hb = make_translation_homography(256, 0);
    matrix Mtest;
    // TODO: fill in RANSAC algorithm.
    for (i = 0; i < k; i++) // for k iterations:
    {
        randomize_matches(m, n);          // shuffle the matches
        Mtest = compute_homography(m, 4); // compute a homography with a few matches (how many??)
        if (!Mtest.data)
        {
            continue; // If homography is empty matrix, continue
        }
        else
        {
            e = model_inliers(Mtest, m, n, thresh);
            if (e > best) //  if new homography is better than old (how can you tell?): (more inliers)
            {
                Hb = compute_homography(m, e); // compute updated homography using all inliers (e), update
                best = e;                      // remember it and how good it is
                if (e > cutoff)                // if it's better than the cutoff:
                {
                    return Hb; // return it immediately
                }
            }
        }
    }
    // if we get to the end return the best homography
    return Hb;
}

// Stitches two images together using a projective transformation.
// image a, b: images to stitch.
// matrix H: homography from image a coordinates to image b coordinates.
// returns: combined image stitched together.
image combine_images(image a, image b, matrix H)
{
    matrix Hinv = matrix_invert(H);

    // Project the corners of image b into image a coordinates.
    point c1 = project_point(Hinv, make_point(0, 0));
    point c2 = project_point(Hinv, make_point(b.w - 1, 0));
    point c3 = project_point(Hinv, make_point(0, b.h - 1));
    point c4 = project_point(Hinv, make_point(b.w - 1, b.h - 1));

    // Find top left and bottom right corners of image b warped into image a.
    point topleft, botright;
    botright.x = MAX(c1.x, MAX(c2.x, MAX(c3.x, c4.x)));
    botright.y = MAX(c1.y, MAX(c2.y, MAX(c3.y, c4.y)));
    topleft.x = MIN(c1.x, MIN(c2.x, MIN(c3.x, c4.x)));
    topleft.y = MIN(c1.y, MIN(c2.y, MIN(c3.y, c4.y)));

    // Find how big our new image should be and the offsets from image a.
    int dx = MIN(0, topleft.x);
    int dy = MIN(0, topleft.y);
    int w = MAX(a.w, botright.x) - dx;
    int h = MAX(a.h, botright.y) - dy;

    // Can disable this if you are making very big panoramas.
    // Usually this means there was an error in calculating H.
    if (w > 7000 || h > 7000)
    {
        fprintf(stderr, "output too big, stopping\n");
        return copy_image(a);
    }

    int i, j, k;
    image c = make_image(w, h, a.c);

    // Paste image a into the new image offset by dx and dy.
    for (k = 0; k < a.c; ++k)
    {
        for (j = 0; j < a.h; ++j)
        {
            for (i = 0; i < a.w; ++i)
            {
                // TODO: fill in.
                set_pixel(c, i, j, k, get_pixel(a, i - dx, j - dy, k));
            }
        }
    }

    // TODO: Paste in image b as well.
    // You should loop over some points in the new image (which? all?)
    // and see if their projection from a coordinates to b coordinates falls
    // inside of the bounds of image b. If so, use bilinear interpolation to
    // estimate the value of b at that projection, then fill in image c.
    point test;
    float val;
    for (k = 0; k < a.c; ++k)
    {
        for (j = 0; j < c4.y; ++j) // Only looping over points within projected bounds
        {
            for (i = 0; i < c4.x; ++i)
            {
                test = project_point(Hinv, make_point(i, j)); // Project to b using given homography
                val = bilinear_interpolate(b, i, j, k);
                set_pixel(c, i - dx, j - dy, k, val);
            }
            // Assign value to c after proper offset
        }
    }

    return c;
}

// Create a panoramam between two images.
// image a, b: images to stitch together.
// float sigma: gaussian for harris corner detector. Typical: 2
// float thresh: threshold for corner/no corner. Typical: 1-5
// int nms: window to perform nms on. Typical: 3
// float inlier_thresh: threshold for RANSAC inliers. Typical: 2-5
// int iters: number of RANSAC iterations. Typical: 1,000-50,000
// int cutoff: RANSAC inlier cutoff. Typical: 10-100
// int draw: flag to draw inliers.
image panorama_image(image a, image b, float sigma, float thresh, int nms, float inlier_thresh, int iters, int cutoff, int draw)
{
    srand(10);
    int an = 0;
    int bn = 0;
    int mn = 0;

    // Calculate corners and descriptors
    descriptor *ad = harris_corner_detector(a, sigma, thresh, nms, &an);
    descriptor *bd = harris_corner_detector(b, sigma, thresh, nms, &bn);

    // Find matches
    match *m = match_descriptors(ad, an, bd, bn, &mn);

    // Run RANSAC to find the homography
    matrix H = RANSAC(m, mn, inlier_thresh, iters, cutoff);

    if (draw)
    {
        // Mark corners and matches between images
        mark_corners(a, ad, an);
        mark_corners(b, bd, bn);
        image inlier_matches = draw_inliers(a, b, H, m, mn, inlier_thresh);
        save_image(inlier_matches, "output/inliers");
    }

    free_descriptors(ad, an);
    free_descriptors(bd, bn);
    free(m);

    // Stitch the images together with the homography
    image comb = combine_images(a, b, H);
    return comb;
}

// Project an image onto a cylinder.
// image im: image to project.
// float f: focal length used to take image (in pixels).
// returns: image projected onto cylinder, then flattened.
image cylindrical_project(image im, float f)
{
    //TODO: project image onto a cylinder

    image c = copy_image(im);
    int xc = (int)(c.w / 2); // centers of projection
    int yc = (int)(c.h / 2);
    float theta, h, xp, yp, zp, xres, yres;
    int i, j, k;
    for (k = 0; k < c.c; ++k)
    {
        for (j = 0; j < c.h; ++j)
        {
            for (i = 0; i < c.w; ++i)
            {
                theta = (i - xc) / f;
                h = (j - yc) / f;
                xp = sinf(theta);
                yp = h;
                zp = cosf(theta);

                xres = f * (xp / zp) + xc;
                yres = f * (yp / zp) + yc;
                set_pixel(c, i, j, k, get_pixel(im, xres, yres, k));
            }
        }
    }

    return c;
}
