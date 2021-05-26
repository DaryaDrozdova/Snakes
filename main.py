import numpy as np
import matplotlib.pyplot as plt
import scipy 
import skimage
from skimage import io, filters
from skimage.util import img_as_float
from scipy import interpolate
import utils
import argparse
from collections import deque

def command_line_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('input_image', type=str)
    parser.add_argument('initial_snake', type=str)
    parser.add_argument('output_image', type=str)
    parser.add_argument('alpha', type=float)
    parser.add_argument('beta', type=float)
    parser.add_argument('tau', type=float)
    parser.add_argument('w_line', type=float)
    parser.add_argument('w_edge', type=float)
    parser.add_argument('kappa', type=float)
    return parser

def IoU(img1, img2):
    overlap = 0
    union = 0
    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            if (img1[i][j] or img2[i][j]):
                union += 1
                if (img1[i][j] and img2[i][j]):
                    overlap += 1
    return overlap / union

def reparametrization(snake):
    N = snake.shape[0]
    snake_len = np.zeros(N)
    length = 0
    for i in range(N):
        length += np.linalg.norm(snake[i] - snake[i - 1])
        snake_len[i] = length
    interpolator_x = interpolate.interp1d(snake_len, snake[:, 0])
    interpolator_y = interpolate.interp1d(snake_len, snake[:, 1])
    for i in range(N - 1):
        snake[i][0] = interpolator_x(i * length / N)
        snake[i][1] = interpolator_y(i * length / N)
    return snake

def build_ext_energy_mat(image, w_line, w_edge):
    edge = filters.sobel(image)
    ext_energy = w_line * image + w_edge * edge
    return ext_energy

def build_int_energy_mat(N, alpha, beta, tau):
    a = np.roll(np.eye(N), -1, axis=0) + np.roll(np.eye(N), -1, axis=1) - 2 * np.eye(N)
    b = np.roll(np.eye(N), -2, axis=0) + np.roll(np.eye(N), -2, axis=1) - 4 * np.roll(np.eye(N), -1, axis=0) - \
        4 * np.roll(np.eye(N), -1, axis=1) + 6 * np.eye(N)
    A = -alpha * a + beta * b
    inverse_A = np.linalg.inv(A + tau * np.eye(N))
    return inverse_A

def build_snake(image, initial_snake, alpha, beta, tau, w_line, w_edge, kappa):
    N = initial_snake.shape[0]
    x, y = initial_snake[:, 0], initial_snake[:, 1]
    ext_matrix = build_ext_energy_mat(image, w_line, w_edge)
    int_matrix = build_int_energy_mat(N, alpha, beta, tau)
    
    interp_img = interpolate.RectBivariateSpline(np.arange(image.shape[1]), np.arange(image.shape[0]), image.T, kx=2, ky=2, s=0)
 
    x_norm = np.zeros(x.shape)
    y_norm = np.zeros(y.shape)

    prev_snake = deque()

    for i in range(200):

        for k in range(N - 1):
                x_norm[k] = (y[k + 1] - y[k])
                y_norm[k] = (x[k] - x[k + 1])
         
        fx = interp_img(x, y, dx=1, grid=False)
        fy = interp_img(x, y, dy=1, grid=False)     
        fx /= np.linalg.norm(fx)
        fy /= np.linalg.norm(fy)
            
        xn = int_matrix @ (tau*x + fx + kappa * x_norm)
        yn = int_matrix @ (tau*y + fy + kappa * y_norm)

        dx = xn - x
        dy = yn - y
        x += dx
        y += dy
        x[-1] = x[0]
        y[-1] = y[0]

        snake = reparametrization(np.stack([y, x], axis=1))
        snake[snake < 0] = 0
        snake[snake[:, 0] > image.shape[1] - 1, 0] = image.shape[1] - 1
        snake[snake[:, 1] > image.shape[0] - 1, 1] = image.shape[0] - 1
        x = snake[:, 1]
        y = snake[:, 0]

        if len(prev_snake) >= 20:
            prev_snake.popleft()

        min_dist = N * 100
        for pair in prev_snake:
            cur_dist = np.average(np.abs(pair[0] - x) + np.abs(pair[1] - y))
            if cur_dist < min_dist:
                min_dist = cur_dist
        prev_snake.append([x.copy(), y.copy()])

        if min_dist < 0.1:
            break
    return np.stack([x, y], axis=1)
   

parser = command_line_parser()
args = parser.parse_args()

image = io.imread(args.input_image)
initial_snake = np.loadtxt(args.initial_snake)[: -1]
gauss_img = filters.gaussian(image, 3)
snake = build_snake(gauss_img, initial_snake, args.alpha, args.beta, args.tau, args.w_line, args.w_edge, args.kappa)

utils.save_mask(args.output_image, snake, image)
