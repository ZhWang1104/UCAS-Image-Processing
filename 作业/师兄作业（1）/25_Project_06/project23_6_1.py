# 课程名称: 图像处理
# 工程作业：06
# 组内学生姓名和学号：
# 完成日期：2023-11-17

# coding: utf-8
import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import math
import cv2


# 预处理函数，接收一个图像src，返回该图的灰度图ndarray
def preprocess(image_src):
    # 读取图像
    image = cv2.imread(image_src)
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# fspecial_motion 函数接收三个参数
# 第一个参数为type,决定该函数的作用，可选参数为motion,directInverse,gaussian,restrictedRangeInverse,
# constantValue_Wiener,constrainedMinimumMeanSquareErrorAlgorithm
# 第二个参数为kernel的矩阵
# 第三个参数为旋转角度
# 返回模糊后的图像的ndarray

def fspecial_motion(image, kernel=12, angle=45):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵，kernel越大，模糊程度越高
    # 第一个参数旋转中心    第二个参数旋转角度    第三个参数旋转后的缩放比例
    M = cv2.getRotationMatrix2D((kernel / 2, kernel / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel))

    # 根据旋转角度和旋转中心获取旋转矩阵
    # 根据旋转矩阵进行仿射变换，即可实现任意角度和任意中心的旋转效果
    motion_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel, kernel))
    motion_kernel = motion_kernel / kernel
    # motion_kernel= np.ones((kernel, kernel)) / (kernel ** 2)
    # 应用滤波器
    result = cv2.filter2D(image, -1, motion_kernel)
    return result, motion_kernel


# 直接逆滤波
# 接收一个ndarray的模糊后的图像，和生成该图像的模糊核kernel
# 返回一个ndarray的复原后的图像

def inverse_filter(blurred_image, kernel):
    # 使用傅里叶变换获取频域表示
    f_blurred = np.fft.fft2(blurred_image)
    f_kernel = np.fft.fft2(kernel, s=blurred_image.shape)

    # 直接逆滤波
    f_result = np.divide(f_blurred, f_kernel)

    # 反变换回时域
    result = np.abs(np.fft.ifft2(f_result))

    return result


# 添加高斯噪声函数
# 接收一个ndarray的模糊后的图像
# 返回一个ndarray的添加高斯噪声后的图像
def add_gaussian_noise(image, mean=0, std=25):
    h, w = image.shape
    noise = np.random.normal(mean, std, (h, w))
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


# 受限范围逆滤波
# 接收一个ndarray的图像，和生成该图像的模糊核kernel，以及估计噪声方差
# 返回一个ndarray的复原后的图像
def wiener_filter(blurred_image, kernel, noise_var):
    # 使用傅里叶变换获取频域表示
    f_blurred = np.fft.fft2(blurred_image)
    f_kernel = np.fft.fft2(kernel, s=blurred_image.shape)

    # 计算Wiener滤波器
    wiener_filter = np.conj(f_kernel) / (np.abs(f_kernel) ** 2 + noise_var)

    # 应用Wiener滤波器
    f_result = f_blurred * wiener_filter

    # 反变换回时域
    result = np.abs(np.fft.ifft2(f_result))

    return result

# 常数值维纳滤波器
# 接收一个ndarray的图像，和生成该图像的模糊核kernel，以及估计噪声方差
# 返回一个ndarray的复原后的图像

def constant_wiener_filter(blurred_image, kernel, noise_var):
    # 使用傅里叶变换获取频域表示
    f_blurred = np.fft.fft2(blurred_image)
    f_kernel = np.fft.fft2(kernel, s=blurred_image.shape)

    # 计算常数值维纳滤波器
    constant_wiener_filter = np.conj(f_kernel) / (np.abs(f_kernel) ** 2 + noise_var)

    # 应用常数值维纳滤波器
    f_result = f_blurred * constant_wiener_filter

    # 反变换回时域
    result = np.abs(np.fft.ifft2(f_result))

    return result

# 约束最小均方误差算法
# 接收一个ndarray的图像，和生成该图像的模糊核kernel，以及估计噪声方差
# 返回一个ndarray的复原后的图像
def constrained_least_squares(blurred_image, kernel, noise_var, alpha=1e-3):
    # 使用傅里叶变换获取频域表示
    f_blurred = np.fft.fft2(blurred_image)
    f_kernel = np.fft.fft2(kernel, s=blurred_image.shape)

    # 计算 CLS 滤波器
    cls_filter = np.conj(f_kernel) / (np.abs(f_kernel) ** 2 + alpha * noise_var)

    # 应用 CLS 滤波器
    f_result = f_blurred * cls_filter

    # 反变换回时域
    result = np.abs(np.fft.ifft2(f_result))

    return result


# 调用cv2显示图像
# 接收一个string类型图像名称和一个ndarray类型的图像
def show(name, image):
    # 显示图像
    cv2.imshow(name, np.uint8(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_src = './th.jpg'
    # 预处理
    image = preprocess(image_src)
    # 应用运动模糊
    blurred, motion_kernel = fspecial_motion(image)

    # 显示原始图像
    show('image', image)
    # 显示模糊后的图像
    show('blurred_image', blurred)
    # 应用直接逆滤波，复原图像
    image_blurred_inverse = inverse_filter(blurred, motion_kernel)
    show('image_blurred_inverse', image_blurred_inverse)

    noisy_image = add_gaussian_noise(blurred)

    # 估计噪声方差（可以根据实际情况调整）
    noise_var = np.var(noisy_image - blurred)

    # 使用直接逆滤波进行图像恢复
    image_blurred_noisy_inverse = inverse_filter(noisy_image, motion_kernel)
    # 使用Wiener滤波进行图像恢复
    restored_image_wiener = wiener_filter(noisy_image, motion_kernel, noise_var)

    show('image_blurred_noisy_inverse', np.uint8(image_blurred_noisy_inverse))
    show('Restored Image (Wiener Filter)', np.uint8(restored_image_wiener))

    noise_variances = [10, 50, 100]
    for noise_var in noise_variances:
        # 添加高斯噪声
        noisy_image = add_gaussian_noise(blurred, std=np.sqrt(noise_var))

        show(f'Noisy Image (Var={noise_var})', noisy_image)

        # 使用常数值维纳滤波进行图像恢复
        restored_image_constant_wiener = constant_wiener_filter(noisy_image, motion_kernel, noise_var)
        show(f'Restored Image (Constant Wiener, Var={noise_var})', np.uint8(restored_image_constant_wiener))

    restored_image_cls = constrained_least_squares(noisy_image, motion_kernel, noise_var)
    show('Restored Image (Constrained Least Squares)', np.uint8(restored_image_cls))
