// aim to update target_matrix_g according to warp_aft_coef (with cubic interpolation)
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// Keys cubic kernel (a = -0.5, Catmull-Rom)
static inline double cubic_weight(double t, double a) {
    double x = t < 0 ? -t : t;   // fabs，inline
    double x2 = x * x;
    double x3 = x2 * x;
    if (x < 1.0)
        return (a + 2)*x3 - (a + 3)*x2 + 1;
    else if (x < 2.0)
        return a*x3 - 5*a*x2 + 8*a*x - 4*a;
    return 0.0;
}

// constrain boundary
inline int clamp(int v, int low_limit, int high_limit) {
    if (v < low_limit) return low_limit;
    if (v > high_limit) return high_limit;
    return v;
}

// Bicubic interpolation
double get_bicubic_interp_value(double *img, int width, int height, double x, double y) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    double a = -0.5; // Catmull-Rom
    double result = 0.0;

    double wx[4], wy[4];

    double dx = x - x0;
    double dy = y - y0;

    for (int i = -1; i <= 2; i++) {
        wx[i + 1] = cubic_weight(dx - i, a);
        wy[i + 1] = cubic_weight(dy - i, a);
    }

    for (int m = -1; m <= 2; m++) {
        int yy = clamp(y0 + m, 0, height - 1);  // y0+m=-1 , 0 result will be same
        double wy_val = wy[m + 1];

        double *row = img + yy * width;

        for (int n = -1; n <= 2; n++) {
            int xx = clamp(x0 + n, 0, width - 1);
            double wx_val = wx[n + 1];
            result += row[xx] * wx_val * wy_val;
        }
    }
    if (result < 0) result = 0;
    if (result > 255) result = 255;
    return result;
}

// aim to update target_matrix_g according to warp_aft_coef
__declspec(dllexport)
void update_target_img_subset_core(
    double *img, double *target_matrix_g, double point_ini[2], double warp_aft_coef[][3],
    int width, int height, int size
){
    int size_half = (int)(0.5*(size-1));
    double local_warp_x, local_warp_y, img_warp_x, img_warp_y;
    for (int y = 0; y < size; y++){
        for (int x = 0; x < size; x++){
            double lx = x - size_half;
            double ly = y - size_half;
            local_warp_x = warp_aft_coef[0][0]*lx + warp_aft_coef[0][1]*ly + warp_aft_coef[0][2];
            local_warp_y = warp_aft_coef[1][0]*lx + warp_aft_coef[1][1]*ly + warp_aft_coef[1][2];
            img_warp_x = point_ini[0] + local_warp_x;
            img_warp_y = point_ini[1] + local_warp_y;
            target_matrix_g[y * size + x] =\
             get_bicubic_interp_value(img, width, height, img_warp_x, img_warp_y);
        }
    }
    return;
}
