#include <math.h>

int interp_safe_margin() {
    return 1;  // bilinear 需要周圍 1 個像素
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

    // 3. 計算目標點與左上角點的距離 (權重)
    double dx = x - x1;
    double dy = y - y1;

    double q11 = data[y1 * width + x1];
    double q21 = data[y1 * width + x2];
    double q12 = data[y2 * width + x1];
    double q22 = data[y2 * width + x2];

    // 先在 X 方向做兩次線性插值，最後在 Y 方向做一次
    double r1 = (1.0f - dx) * q11 + dx * q21;
    double r2 = (1.0f - dx) * q12 + dx * q22;
    double result = (1.0f - dy) * r1 + dy * r2;

    return result;
}