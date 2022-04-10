#include <cuda_runtime.h>

typedef unsigned char uint8_t; // 1个字节8位，存0~255

#define min(a, b) ((a) < (b) ? (a) : (b))

struct Size
{
    int width = 0, height = 0;
    Size() = default;
    Size(int w, int h) : width(w), height(h){};
};

struct AffineMat
{
    // 仿射变换齐次方程式第三行是0,0,1固定,可不存
    float i2d[6]; // input pixel to dst pixel
    float d2i[6];

    void invertAffineTransform(float *imat, float *omat)
    {
        float i00 = imat[0];
        float i01 = imat[1];
        float i02 = imat[2];
        float i10 = imat[3];
        float i11 = imat[4];
        float i12 = imat[5];

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;
        omat[1] = A12;
        omat[2] = b1;
        omat[3] = A21;
        omat[4] = A22;
        omat[5] = b2;
    }

    void compute(const Size &from, const Size &to)
    {
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;
        float scale = min(scale_x, scale_y);
        /*
        第一次平移，from图中心点移动到坐标轴原点，变换矩阵T1
        1    0    -from.width*0.5
        0    1    -from.height*0.5
        0    0    1

        尺度变换，变换矩阵S
        scale    0         0
        0        scale     0
        0        0         1

        第一次平移，将尺度变换后图中心点平移至to图像中心点，变换矩阵T2
        1    0    to.width*0.5
        0    1    to.height*0.5
        0    0    1

        整体变换矩阵M=T2ST1为：
        scale       0         to.width*0.5
        0           scale     to.height*0.5
        0           0         1

        scale       0         -from.width*0.5*scale+to.width*0.5
        0           scale     -from.height*0.5*scale+to.height*0.5
        0           0         1

        中心对齐
        scale       0         -from.width*0.5*scale+to.width*0.5+ scale * 0.5 - 0.5
        0           scale     -from.height*0.5*scale+to.height*0.5+ scale * 0.5 - 0.5
        0           0         1

         + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
         参考：https://www.iteye.com/blog/handspeaker-1545126
        */
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_point(float *mat, int dst_x, int dst_y, float *src_x, float *src_y)
{
    *src_x = mat[0] * dst_x + mat[1] * dst_y + mat[2];
    *src_y = mat[3] * dst_x + mat[4] * dst_y + mat[5];
}

__global__ void warp_affine_kernel(uint8_t *src_device, size_t src_width, size_t src_height, size_t src_line_size,
                                   uint8_t *dst_device, size_t dst_width, size_t dst_height, size_t dst_line_size,
                                   uint8_t fill_value, AffineMat affine_mat)
{
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height)
        return;

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0;
    float src_y = 0;
    affine_point(affine_mat.d2i, dst_x, dst_y, &src_x, &src_y);

    if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
    {
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
    }
    else
    {
        uint8_t const_values[] = {fill_value, fill_value, fill_value};

        int min_x = floorf(src_x);
        int max_x = min_x + 1;
        int min_y = floorf(src_y);
        int max_y = min_y + 1;

        // 双线性插值的四块区域面积
        float a0 = (max_y - src_y) * (max_x - src_x);
        float a1 = (max_y - src_y) * (src_x - min_x);
        float a2 = (src_y - min_y) * (src_x - min_x);
        float a3 = (src_y - min_y) * (max_x - src_x);

        uint8_t *v0 = const_values;
        uint8_t *v1 = const_values;
        uint8_t *v2 = const_values;
        uint8_t *v3 = const_values;

        if (min_y >= 0)
        {
            if (min_x >= 0)
                v0 = src_device + min_y * src_line_size + min_x * 3;

            if (max_x < src_width)
                v1 = src_device + min_y * src_line_size + max_x * 3;
        }

        if (max_y < src_height)
        {
            if (min_x >= 0)
                v2 = src_device + max_y * src_line_size + min_x * 3;

            if (max_x < src_width)
                v3 = src_device + max_y * src_line_size + max_x * 3;
        }
        c0 = floorf(a0 * v0[0] + a1 * v1[0] + a2 * v2[0] + a3 * v3[0] + 0.5f);
        c1 = floorf(a0 * v0[1] + a1 * v1[1] + a2 * v2[1] + a3 * v3[1] + 0.5f);
        c2 = floorf(a0 * v0[2] + a1 * v1[2] + a2 * v2[2] + a3 * v3[2] + 0.5f);
    }
    uint8_t *pdst = dst_device + dst_y * dst_line_size + dst_x * 3;
    pdst[0] = c0;
    pdst[1] = c1;
    pdst[2] = c2;
}

void warp_affine_cuda(uint8_t *src_device, size_t src_width, size_t src_height, size_t src_line_size,
                      uint8_t *dst_device, size_t dst_width, size_t dst_height, size_t dst_line_size,
                      uint8_t fill_value)
{
    AffineMat affine_mat;
    affine_mat.compute(Size(src_width, src_height), Size(dst_width, dst_height));
    dim3 threadOfBlock(32, 32);
    dim3 blocksOfGrid((dst_width + 31) / 32, (dst_height + 31) / 32);
    warp_affine_kernel<<<blocksOfGrid, threadOfBlock, 0, nullptr>>>(src_device, src_width, src_height, src_line_size,
                                                                    dst_device, dst_width, dst_height, dst_line_size,
                                                                    fill_value, affine_mat);
};
