

import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)

py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024)
handlers = py_log_always_on.file_handler_setup(MY_LOGGER)






import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
# arg path -p
parser.add_argument('-p', '--path', type=str, help='path to the data folder. e.g. segnet_150_models_errors_sclera   ')

args = parser.parse_args()
path = Path(args.path)

csv_path = path / '1.csv'
original_df = pd.read_csv(csv_path)

csv_path = path / 'best_0-25_IoU.csv'
iou_alphas_df = pd.read_csv(csv_path)

csv_path = path / 'best_0-25_F1.csv'
f1_alphas_df = pd.read_csv(csv_path)







# csv = csv[['pruning_method', 'alpha', 'retained_flops', 'val IoU', 'val F1', 'test IoU', 'test F1']]



# make the csv have a multiindex that is then easier to work with:

def set_multiindex(df, column_names):
    df.set_index(column_names, inplace=True)
    df.index = pd.MultiIndex.from_frame(df.index.to_frame())    # this was for google sheets approach:  .fillna(method='ffill'))  # Fill the NaN values from merged cells in the index
    df.sort_index(inplace=True)
    return df

set_multiindex(original_df, ['pruning_method', 'best_alpha', 'best_value'])

set_multiindex(iou_alphas_df, ['pruning_method'])
set_multiindex(f1_alphas_df, ['pruning_method'])




















# create empty df
columns = ['pruning_method', 'best_alpha', 'best_value']
df_iou = pd.DataFrame(columns=columns)
df_f1 = pd.DataFrame(columns=columns)


# Example of how to use the multiindex:

for pr_method in df.index.unique(level='pruning_method'):

    if pr_method == "no_pruning":
        continue


    df_0 = df.loc[pr_method]
    # cross selecting for retained_flops = 0.25
    df_0 = df_0.xs(0.25, level='retained_flops')


    df_0_iou = df_0['val IoU']
    df_0_f1 = df_0['val F1']

    # get best iou in this df
    best_iou = df_0_iou.max()
    best_alpha_iou = df_0_iou.idxmax()

    best_f1 = df_0_f1.max()
    best_alpha_f1 = df_0_f1.idxmax()

    new_row_iou = pd.DataFrame({'pruning_method': [pr_method], 'best_alpha': [best_alpha_iou], 'best_value': [best_iou]})
    new_row_f1 = pd.DataFrame({'pruning_method': [pr_method], 'best_alpha': [best_alpha_f1], 'best_value': [best_f1]})

    df_iou = pd.concat([df_iou, new_row_iou], ignore_index=True)
    df_f1 = pd.concat([df_f1, new_row_f1], ignore_index=True)




to_iou_csv = path / 'best_0.25_IoU.csv'
to_f1_csv = path / 'best_0.25_F1.csv'

df_iou.to_csv(to_iou_csv)
df_f1.to_csv(to_f1_csv)


sys.exit(0)












import math
from matej.collections import DotDict, ensure_iterable
from matej.colour import truncate_colourmap

CMAP = truncate_colourmap(plt.cm.plasma_r, start=.2)  # Colourmap to use in figures
FIG_EXTS = 'pdf',# 'png', 'svg', 'eps'  # Which formats to save figures to

FIG_EXTS = ensure_iterable(FIG_EXTS, str)
colourise = lambda x: zip(x, CMAP(np.linspace(0, 1, len(x))))
large_fmt = lambda x, _=None: np.format_float_positional(x, precision=3, fractional=False, trim='-')  # For values that can go over absolute value of 1

small_fmt = lambda x, _=None: np.format_float_positional(x, precision=3, trim='-')  # For values with absolute value between 0 and 1
percent_fmt = lambda x, _=None: large_fmt(100 * x) + "%"
convert_percent = lambda x: float(x.rstrip('%')) / 100 if isinstance(x, str) else float(x)
oom = lambda x: 10 ** math.floor(math.log10(x))  # Order of magnitude: oom(0.9) = 0.1, oom(30) = 10

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

class Figure(ABC):
	def __init__(self, name, save_dir, *, fontsize=24):
		self.name = name
		self.dir = save_dir/type(self).__name__
		self.fontsize = fontsize
		self.fig = None
		self.ax = None

	@abstractmethod
	def __enter__(self, *args, **kw):
		plt.rcParams['font.size'] = self.fontsize
		self.fig, self.ax = plt.subplots(*args, num=str(self.dir/self.name), **kw)
		return self

	def __exit__(self, *args, **kw):
		plt.rcParams['font.size'] = self.fontsize
		self.close()
		plt.close(self.fig)

	@abstractmethod
	def close(self):
		pass

	@abstractmethod
	def plot(self):
		plt.rcParams['font.size'] = self.fontsize

	def save(self, name=None, fig=None):
		if fig is None:
			fig = self.fig
		if name is None:
			name = self.name
		self.dir.mkdir(parents=True, exist_ok=True)
		for ext in FIG_EXTS:
			save = self.dir/f'{name}.{ext}'
			print(f"Saving to {save}")
			fig.savefig(save, bbox_inches='tight')

	@staticmethod
	def _nice_tick_size(min_, max_, min_ticks=3, max_ticks=7):
		diff = max_ - min_
		return min(
			oom(diff) * np.array([.1, .2, .5, 1, 2, 5]),  # Different possible tick sizes
			key=lambda tick_size: (max(0, min_ticks - (n_ticks := diff // tick_size + 1), n_ticks - max_ticks), n_ticks)  # Return the one closest to the requested number of ticks. If several are in the range, return the one with the fewest ticks.
		)

	def horizontal_line(self, *args, **kw):
		return self.ax.axhline(*args, **kw)

	def vertical_line(self, *args, **kw):
		return self.ax.axvline(*args, **kw)


class ScatterLine(Figure):
	def __init__(self, name, save_dir, *, xlabel="FLOPs", ylabel="IoU", xticklabels=None, fontsize=28):
		super().__init__(name, save_dir, fontsize=fontsize)
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.xticklabels = xticklabels
		self.y = None

	def __enter__(self):
		super().__enter__()
		self.ax.grid(axis='y', which='major', alpha=.5)
		self.ax.grid(axis='y', which='minor', alpha=.2)
		self.ax.xaxis.set_major_formatter(FuncFormatter(percent_fmt))
		self.ax.yaxis.set_major_formatter(FuncFormatter(small_fmt))
		self.ax.margins(0)
		self.ax.set_xlabel(self.xlabel)
		self.ax.set_ylabel(self.ylabel)
		self.fig.tight_layout(pad=0)
		self.y = []
		return self

	def close(self, *args, **kw):
		ymin = min(self.y) if self.y else 0
		ymax = max(self.y) if self.y else 1

		if self.xticklabels is not None:
			self.ax.set_xticks(np.arange(len(self.xticklabels)))
			self.ax.set_xticklabels(self.xticklabels)

		ytick_size = self._nice_tick_size(ymin, ymax)
		ymin = max(0, ymin - ytick_size)
		ymax += ytick_size
		self.ax.yaxis.set_major_locator(MultipleLocator(ytick_size))
		self.ax.yaxis.set_minor_locator(MultipleLocator(ytick_size / 2))

		self.ax.set_xlim(self.ax.get_xlim()[0] - .2, self.ax.get_xlim()[1] + .2)
		self.ax.set_ylim(ymin, ymax)
		self.save(f'{self.name} (No Legend)')
		if self.ax.get_legend_handles_labels()[0]:
			self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
		self.save()

	def plot(self, y, *, label=None, color=None):
		super().plot()
		y = ensure_iterable(y)
		if self.xticklabels is not None and len(self.xticklabels) != len(y):
			raise ValueError(f"Length of plot ({len(y)}) should equal number of x ticks ({len(self.xticklabels)})")

		marker = 'x' if len(y) > 1 and any(val != y[0] for val in y[1:]) else ''

		self.ax.plot(np.arange(len(y)), y, '-' + marker, markersize=12, linewidth=2, label=label, color=color)

		self.y.extend(y)


class Bar(Figure):
	def __init__(self, name, save_dir, groups, n=1, *, ylabel="IoU", fontsize=32, margin=.2):
		super().__init__(name, save_dir, fontsize=fontsize)
		self.groups = groups
		self.m = len(groups)
		self.n = n
		self.ylabel = ylabel
		self.margin = margin
		self.width = (1 - self.margin) / self.n
		self.min = None
		self.max = None

	def __enter__(self):
		super().__enter__(figsize=(15, 5))
		self.ax.grid(axis='y', which='major', alpha=.5)
		self.ax.grid(axis='y', which='minor', alpha=.2)
		self.ax.margins(0)
		self.ax.yaxis.set_major_formatter(FuncFormatter(small_fmt))
		self.ax.set_ylabel(self.ylabel)
		self.fig.tight_layout(pad=0)
		self.min = float('inf')
		self.max = float('-inf')
		return self

	def close(self, *args, **kw):
		handles, labels = self.ax.get_legend_handles_labels()
		if labels:
			by_label = dict(zip(labels, handles))  # Remove duplicate labels
			for attempt in range(4, 1, -1):
				if len(by_label) % attempt == 0:
					ncol = attempt
					break
			else:
				ncol = 3
			self.ax.legend(by_label.values(), by_label.keys(), ncol=ncol, bbox_to_anchor=(.02, 1.02, .96, .1), loc='lower left', mode='expand', borderaxespad=0)
		self.ax.tick_params(axis='x', pad=10)
		self.ax.set_xticks(np.arange(self.m) + (self.margin + self.n * self.width) / 2)
		self.ax.set_xticklabels(self.groups)  # Default x labels
		# self.ax.set_xticklabels(self.groups, rotation=60, ha='right', rotation_mode='anchor')  # Rotated x labels
		# self.ax.set_xticklabels([group if g % 2 else f"\n{group}" for g, group in enumerate(self.groups)])  # 2-row x labels
		ymin = self.min if self.min != float('inf') else 0
		ymax = self.max if self.max != float('-inf') else 1
		ytick_size = self._nice_tick_size(ymin, ymax)
		ymin = ymin - .5 * ytick_size if ymin < 0 or ymin - .5 * ytick_size >= 0 else 0
		ymax += .5 * ytick_size
		self.ax.set_ylim(ymin, ymax)
		self.ax.yaxis.set_major_locator(MultipleLocator(ytick_size))
		self.ax.yaxis.set_minor_locator(MultipleLocator(ytick_size / 2))
		self.save()
		self.ax.tick_params(axis='x', bottom=False, labelbottom=False)
		#self.save(f'{self.name} (No Labels)')

	def plot(self, val, group=0, index=0, *, std=None, width=1, label=None, color=None):
		super().plot()
		plt.rcParams['font.size'] = 10
		err_w = np.clip(self.width * 10, 2, 5)
		self.ax.bar(
			group + self.margin / 2 + index * (self.width + width - 1),
			val,
			yerr=std,
			error_kw=dict(lw=err_w, capsize=1.5 * err_w, capthick=.5 * err_w),
			width=width * self.width,
			align='edge',
			label=label,
			color=color
		)
		self.min = min(self.min, val - std if std else val)
		self.max = max(self.max, val + std if std else val)


class Heatmap(Figure):
	def __init__(self, name, save_dir, labels=None, *args, xlabels=None, ylabels=None, **kw):
		super().__init__(name, save_dir, *args, **kw)
		self.xlabels = self.ylabels = labels
		if xlabels is not None:
			self.xlabels = xlabels
		if ylabels is not None:
			self.ylabels = ylabels

	def __enter__(self):
		super().__enter__()
		self.ax.margins(0)
		self.fig.tight_layout(pad=0)
		return self

	def close(self, *args, **kw):
		self.ax.set_xticks(range(len(self.xlabels)))
		self.ax.set_yticks(range(len(self.ylabels)))
		self.ax.set_xticklabels(self.xlabels)
		self.ax.set_yticklabels(self.ylabels)
		self.save()

	def plot(self, matrix):
		super().plot()
		p = self.ax.matshow(matrix, vmin=-1, vmax=1, cmap='bwr')
		self.fig.colorbar(p)

