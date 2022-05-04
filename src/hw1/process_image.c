#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    // Padding/Clamping
    x = (x < 0) ? 0 : x;
    x = (x >= im.w) ? im.w - 1 : x;

    y = (y < 0) ? 0 : y;
    y = (y >= im.h) ? im.h - 1 : y;

    c = (c < 0) ? 0 : c;
    c = (c >= im.c) ? im.w - 1 : c;

    // Location = x + y*W + z*W*H
    int loc = x + (y * im.w) + (c * im.w * im.h);
    return im.data[loc];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // TODO Fill this in
    if (x < im.w && x >= 0 &&
        y < im.h && y >= 0 &&
        c < im.c && c >= 0)
    {
        int loc = x + (y * im.w) + (c * im.w * im.h);
        im.data[loc] = v;
    }
    /*
    assert(x < im.w);
    assert(y < im.h);
    assert(c < im.c);
    */
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    // TODO Fill this in
    int w, h, c;
    for (w = 0; w < im.w; w++)
    {
        for (h = 0; h < im.h; h++)
        {
            for (c = 0; c < im.c; c++)
            {
                set_pixel(copy, w, h, c, get_pixel(im, w, h, c));
            }
        }
    }
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    float gray_pixel;
    // TODO Fill this in
    int w, h, c;
    for (w = 0; w < im.w; w++)
    {
        for (h = 0; h < im.h; h++)
        {
            gray_pixel = 0.299 * get_pixel(im, w, h, 0) +
                         0.587 * get_pixel(im, w, h, 1) +
                         0.114 * get_pixel(im, w, h, 2);
            set_pixel(gray, w, h, 0, gray_pixel);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    int w, h;
    for (w = 0; w < im.w; w++)
    {
        for (h = 0; h < im.h; h++)
        {
            set_pixel(im, w, h, c, get_pixel(im, w, h, c) + v);
        }
    }
}

void clamp_image(image im)
{
    // TODO Fill this in
    int w, h, c;
    for (w = 0; w < im.w; w++)
    {
        for (h = 0; h < im.h; h++)
        {
            for (c = 0; c < im.c; c++)
            {
                if (get_pixel(im, w, h, c) < 0)
                {
                    set_pixel(im, w, h, c, 0);
                }
                else if (get_pixel(im, w, h, c) > 1)
                {
                    set_pixel(im, w, h, c, 1);
                }
            }
        }
    }
}

// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    float H, S, V, C, m, H_prime, R, G, B;
    int w, h;
    for (w = 0; w < im.w; w++)
    {
        for (h = 0; h < im.h; h++)
        {
            V = three_way_max(get_pixel(im, w, h, 0),
                              get_pixel(im, w, h, 1),
                              get_pixel(im, w, h, 2));
            if (V == 0)
            {
                S = 0;
            }
            else
            {
                m = three_way_min(get_pixel(im, w, h, 0),
                                  get_pixel(im, w, h, 1),
                                  get_pixel(im, w, h, 2));
                C = V - m;
                S = C / V;
                if (C == 0)
                {
                    H = 0;
                }
                else
                {
                    R = get_pixel(im, w, h, 0);
                    G = get_pixel(im, w, h, 1);
                    B = get_pixel(im, w, h, 2);
                    if (V == R)
                    {
                        H_prime = (G - B) / C;
                    }
                    else if (V == G)
                    {
                        H_prime = ((B - R) / C) + 2;
                    }
                    else /* V == B */
                    {
                        H_prime = ((R - G) / C) + 4;
                    }

                    H = (H_prime / 6);
                    if (H_prime < 0)
                    {
                        H += 1;
                    }
                }
                set_pixel(im, w, h, 0, H); // R -> H
                set_pixel(im, w, h, 1, S); // G -> S
                set_pixel(im, w, h, 2, V); // B -> V
            }
        }
    }
}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    float H, S, V, Hi, F, P, Q, T;
    int w, h;
    for (w = 0; w < im.w; w++)
    {
        for (h = 0; h < im.h; h++)
        {
            H = get_pixel(im, w, h, 0);
            S = get_pixel(im, w, h, 1);
            V = get_pixel(im, w, h, 2);

            H = H * 6;
            Hi = floor(H);
            F = H - Hi;
            P = V * (1 - S);
            Q = V * (1 - F * S);
            T = V * (1 - (1 - F) * S);
            if (Hi == 0)
            {
                set_pixel(im, w, h, 0, V);
                set_pixel(im, w, h, 1, T);
                set_pixel(im, w, h, 2, P);
            }
            else if (Hi == 1)
            {
                set_pixel(im, w, h, 0, Q);
                set_pixel(im, w, h, 1, V);
                set_pixel(im, w, h, 2, P);
            }
            else if (Hi == 2)
            {
                set_pixel(im, w, h, 0, P);
                set_pixel(im, w, h, 1, V);
                set_pixel(im, w, h, 2, T);
            }
            else if (Hi == 3)
            {
                set_pixel(im, w, h, 0, P);
                set_pixel(im, w, h, 1, Q);
                set_pixel(im, w, h, 2, V);
            }
            else if (Hi == 4)
            {
                set_pixel(im, w, h, 0, T);
                set_pixel(im, w, h, 1, P);
                set_pixel(im, w, h, 2, V);
            }
            else /* (Hi == 5) */
            {
                set_pixel(im, w, h, 0, V);
                set_pixel(im, w, h, 1, P);
                set_pixel(im, w, h, 2, Q);
            }
        }
    }
}
