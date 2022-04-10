#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <string>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

void warp_affine_cuda(uint8_t *src_device, size_t src_width, size_t src_height, size_t src_line_size,
                      uint8_t *dst_device, size_t dst_width, size_t dst_height, size_t dst_line_size,
                      uint8_t fill_value); // 声明，定义在cu文件中

void warp_affine(const cv::Mat &src, cv::Mat &dst, uint8_t fill_value = 114)
{

    uint8_t *src_device = nullptr;
    uint8_t *dst_device = nullptr;

    size_t src_size = src.cols * src.rows * 3;
    size_t dst_size = dst.cols * dst.rows * 3;

    // 在GPU上开辟两块空间，内存地址增加顺序是沿通道、再沿行、再沿列
    checkRuntime(cudaMalloc(&src_device, src_size));
    checkRuntime(cudaMalloc(&dst_device, dst_size));
    checkRuntime(cudaMemcpy(src_device, src.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上

    warp_affine_cuda(src_device, src.cols, src.rows, src.cols * 3, dst_device, dst.cols, dst.rows, dst.cols * 3, fill_value);

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(dst.data, dst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(src_device));
    checkRuntime(cudaFree(dst_device));
}

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        printf("require arg: image path");
        exit(EXIT_FAILURE);
    }
    else
    {
        std::string image_path(argv[1]);
        printf("input image path: %s\n", image_path.c_str());
        cv::Mat image = cv::imread(image_path);
        cv::Mat output(cv::Size(640, 640), CV_8UC3);
        warp_affine(image, output);
        cv::imwrite("output.jpg", output);
        printf("Done. save to output.jpg\n");
    }

    return 0;
}