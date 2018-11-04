import matplotlib.pyplot as plt
import math
import time
import sys
import os


euler = lambda x, y, h, yp : y + h * yp(x, y)
improved_euler = lambda x, y, h, yp : y + h * (yp(x, y) + yp(x + h, y + h * yp(x, y))) / 2
runge_kutta = lambda x, y, h, yp : y + h * (yp(x, y) + 2 * yp(x + h / 2, y + h * yp(x, y) / 2) + 2 * yp(x + h / 2, y + h * (yp(x + h / 2, y + h * yp(x, y) / 2)) / 2) + yp(x + h, y + h * yp(x + h / 2, y + h * (yp(x + h / 2, y + h * yp(x, y) / 2)) / 2))) / 6
solve = lambda xs, ys, method, h, yp : list(map(lambda x : ys.append(method(x, ys[-1], h, yp)), xs[:-1]))
error = lambda ys, correct : list(map(lambda y: math.fabs(y[1] - y[0]), zip(ys, correct)))
figsize=(14, 8)


def plot(method, h, yp, func, xs, y0):
	ys = [y0]
	solve(xs, ys, method, h, yp)
	return ys, error(ys, list(map(func, xs)))


def compute(x0, y0, yp, f, c, X, h):
	n = int((X - x0) / h)
	const = c(x0, y0)
	func = lambda x : f(x, const)
	xs = [x0 + h * i for i in range(n + 1)]
	make_plot = lambda method : plot(method, h, yp, func, xs, y0)
	ys_euler, err_euler = make_plot(euler)
	ys_improved_euler, err_improved_euler = make_plot(improved_euler)
	ys_runge_kutta, err_runge_kutta = make_plot(runge_kutta)
	return xs, ys_euler, err_euler, ys_improved_euler, err_improved_euler, ys_runge_kutta, err_runge_kutta


def draw_total_error(total_euler, total_improved_euler, total_runge_kutta, inds, h):
	plt.figure(figsize=figsize)
	ax = plt.gca()
	ax.set_title('Total error')
	width = 0.9
	ax.bar(inds, total_euler,  width=width, color='b', label='Euler')
	ax.bar(inds, total_improved_euler, width=width * 2 / 3, color='g', label='Improved Euler')
	ax.bar(inds, total_runge_kutta, width=width / 3, color='r', label='Runge-Kutta')
	plt.xticks(inds, list(map(lambda i : f'{i * h:.2f}', inds)))
	plt.xlabel('Step size')
	plt.ylabel('Max global error')
	plt.legend()
	plt.savefig('total_error.png')


def draw_plots(xs, ys_exact, ys_euler, ys_improved_euler, ys_runge_kutta):
	plt.figure(figsize=figsize)
	ax = plt.gca()
	ax.set_title('Plots')
	ax.plot(xs, ys_exact, marker='_', color='y', label='Exact')
	ax.plot(xs, ys_euler, marker='o', color='b', label='Euler')
	ax.plot(xs, ys_improved_euler, marker='^', color='g', label='Improved Euler')
	ax.plot(xs, ys_runge_kutta, marker='D', color='r', label='Runge-Kutta')
	plt.legend()
	plt.savefig('plots.png')


def draw_errors(xs, err_euler, err_improved_euler, err_runge_kutta):
	plt.figure(figsize=figsize)
	ax = plt.gca()
	ax.set_title('Global errors')
	ax.plot(xs, err_euler, marker='o', color='b', label='Euler')
	ax.plot(xs, err_improved_euler, marker='^', color='g', label='Improved Euler')
	ax.plot(xs, err_runge_kutta, marker='D', color='r', label='Runge-Kutta')
	plt.legend()
	plt.savefig('global_errors.png')


def change_step(compute_with_new_step, start_step, inds):
	total_euler, total_improved_euler, total_runge_kutta = [], [], []
	for i in inds:
		xs, ys_euler, err_euler, ys_improved_euler, err_improved_euler, ys_runge_kutta, err_runge_kutta = compute_with_new_step(start_step * i)
		total_euler.append(max(err_euler))
		total_improved_euler.append(max(err_improved_euler))
		total_runge_kutta.append(max(err_runge_kutta))
	return total_euler, total_improved_euler, total_runge_kutta, inds


def main(x0 = 0, y0 = 0, yp = lambda x, y : 4 * x - 2 * y, f = lambda x, c: c * math.exp(-2 * x) + 2 * x - 1, c = lambda x0, y0: (y0 + 1 - 2 * x0) * math.exp(2 * x0), X = 3, h = 0.15, bars = 11):
	xs, ys_euler, err_euler, ys_improved_euler, err_improved_euler, ys_runge_kutta, err_runge_kutta = compute(x0, y0, yp, f, c, X, h)
	ys_exact = list(map(lambda x : f(x, c(x0, y0)), xs))
	draw_plots(xs, ys_exact, ys_euler, ys_improved_euler, ys_runge_kutta)
	draw_errors(xs, err_euler, err_improved_euler, err_runge_kutta)
	start_step = h / ((bars // 2) + 1)
	total_euler, total_improved_euler, total_runge_kutta, inds = change_step(lambda h : compute(x0, y0, yp, f, c, X, h), start_step, range(1, bars + 1))
	draw_total_error(total_euler, total_improved_euler, total_runge_kutta, inds, start_step)

if __name__ == '__main__':
	dir_name = str(int(time.time()))
	if not os.path.exists(dir_name):
		try:
			os.makedirs(dir_name)
			os.chdir(dir_name)
		except:
			sys.exit('Can\'t change working directory')
		main()
	else:
		sys.exit('Can\'t create new directory')
