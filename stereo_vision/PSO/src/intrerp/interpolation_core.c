#include <math.h>

int interp_safe_margin() {
    return 1;  // bilinear need padding 1 pixel
}

double bilinear(double *data, int width, int height, double x, double y) {
    int x1 = (int)x;
    int y1 = (int)y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (x2 >= width) x2 = width - 1;
    if (y2 >= height) y2 = height - 1;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;

    // 3. distance between target pt and left up pt (weight)
    double dx = x - x1;
    double dy = y - y1;

    double q11 = data[y1 * width + x1];
    double q21 = data[y1 * width + x2];
    double q12 = data[y2 * width + x1];
    double q22 = data[y2 * width + x2];

    //  X: 2 times linear interpolation. Y: 1 times
    double r1 = (1.0f - dx) * q11 + dx * q21;
    double r2 = (1.0f - dx) * q12 + dx * q22;
    double result = (1.0f - dy) * r1 + dy * r2;

    return result;
}