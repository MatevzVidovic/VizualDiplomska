#!/usr/bin/env python3

# Always import these
import argparse
from ast import literal_eval
from functools import partial
from matej.argparse import StoreDictPairsAction
from matej.callable import make_module_callable
from matej.collections import DotDict, ensure_iterable
from matej.gui.tkinter import ToolTip
from pathlib import Path
import sys
import textwrap
from tkinter import *
from tkinter.colorchooser import askcolor
import tkinter.filedialog as filedialog

# # If you need EYEZ
# ROOT = Path(__file__).absolute().parent.parent
# if str(ROOT) not in sys.path:
# 	sys.path.append(str(ROOT))
# from eyez.utils import EYEZ

# Import whatever else is needed
from abc import ABC, abstractmethod
import logging
import math
from matej.colour import truncate_colourmap
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd




# f = /asdf/vsdaj/iosad.goi
# fname = iosad.goi
# bname = iosad
# ext = .goi
# dir = /asdf/vsdaj(/)

# If you get "Cannot connect to X display" errors, run this script as:
# MPLBACKEND=Agg python segmentation_evaluation_plot_and_save_ge.py
# matplotlib should handle this automatically but it can fail when $DISPLAY is set but an X-server is not available (such as when connecting remotely with ssh -Y without running an X-server on the client)


# Constants
URL = r'https://docs.google.com/spreadsheets/d/1t91TkyvsHIxzxjopXqrtyuFJ7oW1FRe1G4yNaL4oOH8/export?gid={}&format=csv'
FIG_EXTS = 'pdf',# 'png', 'svg', 'eps'  # Which formats to save figures to
CMAP = truncate_colourmap(plt.cm.plasma_r, start=.2)  # Colourmap to use in figures
MARKERS = 'osP*Xv^<>p1234'
L1, L2 = (fr"$\mathregular{{L^{n+1}}}$" for n in range(2))
METHODS = {
	"L1": L1,
	"L2": L2,
	"L1 LeGR": f"{L1} LeGR",
	"L2 LeGR": f"{L2} LeGR",
	"Uniform": "Uniform",
	"Random": "Random"
}

# Auxiliary stuff
FIG_EXTS = ensure_iterable(FIG_EXTS, str)
colourise = lambda x: zip(x, CMAP(np.linspace(0, 1, len(x))))
large_fmt = lambda x, _=None: np.format_float_positional(x, precision=3, fractional=False, trim='-')  # For values that can go over absolute value of 1
small_fmt = lambda x, _=None: np.format_float_positional(x, precision=3, trim='-')  # For values with absolute value between 0 and 1
percent_fmt = lambda x, _=None: large_fmt(100 * x) + "%"
convert_percent = lambda x: float(x.rstrip('%')) / 100 if isinstance(x, str) else float(x)
oom = lambda x: 10 ** math.floor(math.log10(x))  # Order of magnitude: oom(0.9) = 0.1, oom(30) = 10
logging.getLogger('matplotlib.backends.backend_ps').addFilter(lambda record: 'PostScript backend' not in record.getMessage())  # Suppress matplotlib warnings about .eps transparency


class Main(DotDict):
	def __init__(self):
		#TEMPLATE: Program description. Either a string or a tuple of two strings (one short, one longer)
		self._description = "Template", "Template for scripts that can be run via CLI arguments or through the GUI"

		#TEMPLATE: Automatic arguments. These are added to the argument parser and GUI automatically.
		self._automatic = dict(
			# Path arguments
			# argument=PathArg(default value, help, *filetypes)
			# Default value can be None. In the GUI an empty path means a None path (not the current directory, as is the default in pathlib.Path)
			# Regarding filetypes: if directory, leave empty; if file but you don't want to give an extension list, use None; otherwise see https://stackoverflow.com/a/44403840/5769814
			save=PathArg('Segmentation/Sclera/Results/2023 JKSU-CIS - IPAD', "Path to save the results to")

			# Boolean arguments
			# argument=BoolArg(default value, help, *possible additional argument flags (--argument-name will be added automatically at the start if not present))
			# Note that this removes the ability to combine flags like -of (you have to use -o -f instead)

			# Choice arguments
			# argument=ChoiceArg(default value, (choice1, choice2, ...), help, *argument flags, choice_descriptions=(), type=None)
			# If type not passed (or None), will try to infer (int -> float -> str)

			# Single number arguments
			# argument=NumberArg(default value, (min, max, step), help, *argument flags, type=None, gui_type='slider')
			# If type not passed (or None), will try to infer (int -> float). gui_type can be 'slider' or 'spinbox'

			# Colour arguments
			# argument=ColourArg(default value, help, *argument flags)
		)
		# Add all the above to argument list
		for attr, arg in self._automatic.items():
			self[attr] = arg.default

		#TEMPLATE: Other arguments. You'll have to manually add these to the argument parser and GUI.
		# self.complicated_argument = ["Default", "Value"]

		#TEMPLATE: Which arguments can be passed as varargs (paths are added by default)
		self._varargs = [attr for attr, arg in self._automatic.items() if isinstance(arg, PathArg)]
		# self._varargs.append(argument_name)

		#TEMPLATE: Other instance attributes you want to declare (these should start with _ so they don't get mixed up with the arguments)
		# self._result = None

		# Extra keyword arguments passed in with -e flag or in extra Key-Value frames in the GUI
		self.extra = DotDict()

	def args(self):
		return {k: v for k, v in self.items() if not k.startswith('_')}

	def __str__(self):
		return str(self.args())

	# Support for *-unpacking arguments
	def __iter__(self):
		return iter([self[attr] for attr in self._varargs])

	# Support for **-unpacking arguments
	def keys(self):
		return self.args().keys()

	##############################
	# Main code                  #
	##############################

	def __call__(self):
		#TEMPLATE: Insert your own main code here. You can access the arguments as self.argname
		self._figures = self.save/'Figures'
		self._latex = self.save/'Latex'
		self._figures.mkdir(parents=True, exist_ok=True)
		self._latex.mkdir(parents=True, exist_ok=True)

		plt.rcParams['font.family'] = 'Times New Roman'
		plt.rcParams['font.weight'] = 'normal'
		plt.rcParams['font.size'] = 24



		unet_sclera_df = pd.read_csv(URL.format(1909047693)) #, index_col=(0, 1))

		print(unet_sclera_df)

		if False:

			# Read the dataframes from Google Sheets
			base_df = pd.read_csv(URL.format(2103778960), index_col=(0, 1))
			pruning_df = pd.read_csv(URL.format(995585175), index_col=(0, 1, 2, 3))
			validation_df = pd.read_csv(URL.format(742301031), index_col=(0, 1, 2, 3))
			ablation_df = pd.read_csv(URL.format(1825603444), index_col=(0, 1, 2))
			ablation_validation_df = pd.read_csv(URL.format(1985353682), index_col=(0, 1, 2))

			# Fill the NaN values in Uniform/Random pruning rows
			pruning_df.fillna(method='ffill', axis=1, inplace=True)

			for df in (base_df, pruning_df, validation_df, ablation_df, ablation_validation_df):
				df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna(method='ffill'))  # Fill the NaN values from merged cells in the index
				df.sort_index(inplace=True)
				for col in df:
					df[col] = df[col].apply(convert_percent)  # We have the data formatted as percentages, so convert it into floats

			for dataset in pruning_df.index.levels[0]:
				with (self._latex/f'{dataset}.tex').open('w', encoding='utf-8') as latex:
					ylabel = "mIoU" if dataset == 'MOBIUS' else "IoU"
					for model in pruning_df.index.levels[1]:
						base = base_df['(m)IoU'][dataset, model]

						for alpha in pruning_df:
							df = pruning_df[alpha][dataset, model]
							figname = f'{dataset} - {model} ({alpha.split("=")[1].strip()})'

							# ScatterLine over different FLOPs
							with ScatterLine(figname, self._figures, xlabel="FLOPs", xticklabels=("25%", "50%", "75%", "100%"), ylabel=ylabel) as scline:
								df = df.swaplevel().sort_index(level='Method')
								for (key, method), colour in colourise(METHODS.items()):
									scline.plot(np.append(df[key].values, base), label=method, color=colour)

						for flops in pruning_df.index.levels[2]:
							df = pruning_df.T[dataset, model, flops]
							val_df = validation_df.T[dataset, model, flops]
							df = df.reindex(METHODS.keys(), axis=1)
							val_df = val_df.reindex([method for method in METHODS if method in val_df.keys()], axis=1)

							figname = f'{dataset} - {model} ({flops})'

							# ScatterLine over different αs
							methods = list(METHODS.values())
							with ScatterLine(figname, self._figures, xlabel="α", xticklabels=(0, 0.1, 0.5, 0.9, 1), ylabel=ylabel) as scline:
								block_max = df.values.max()
								for (key, method), colour in colourise(METHODS.items()):
									row = df[key].values
									self._write_latex_line(latex, row, block_max, model, flops, method, *pruning_df.index.levels[1:3], methods)
									scline.plot(row, label=method, color=colour)

							# Bar graph with best α, α=1, and uniform/random/unpruned
							methods += ["Unpruned"]
							# best_results = df[df.index != 'α = 1'].max().tolist()[:4]  # Best α on test data
							best_results = [df[df.index != 'α = 1'].loc[row, col] for col, row in val_df.idxmax().items()]  # Best α on validation data
							baselines = df[df.index == 'α = 1'].values.flatten().tolist()
							baseline_results = baselines[:4]
							other_results = baselines[4:] + [base]
							with Bar(figname, self._figures, groups=methods, n=2, ylabel=ylabel) as bar:
								for i, results in enumerate(zip(best_results, baseline_results)):
									for j, (result, colour, label) in enumerate(zip(results, (CMAP(0), CMAP(.5)), ("IPAD (α = opt.)", "Weights only (α = 1)"))):
										bar.plot(result, i, j, label=label, color=colour)
								for i, result in enumerate(other_results):
									bar.plot(result, i + len(best_results), label="Baseline", color=CMAP(1), width=2)
								scline.plot([base] * 5, label="Unpruned", color='black')

							# Bar graph with best α, α=0, α=0.5, and α=1
							# best_results = df.max().tolist()[:4]  # Best α on test data
							best_results = [df.loc[row, col] for col, row in val_df.idxmax().items()]  # Best α on validation data
							w0_results = df[df.index == 'α = 0'].values.flatten().tolist()[:4]
							w05_results = df[df.index == 'α = 0.5'].values.flatten().tolist()[:4]
							w1_results = df[df.index == 'α = 1'].values.flatten().tolist()[:4]
							with Bar(f'{figname} - Ablation', self._figures, groups=methods[:4], n=4, ylabel=ylabel) as bar:
								for i, results in enumerate(zip(best_results, w0_results, w05_results, w1_results)):
									for (j, (result, colour)), label in zip(enumerate(colourise(results)), ("α = opt.", "α = 0", "α = 0.5", "α = 1")):
										bar.plot(result, i, j, label=label, color=colour)

							if model == 'RITnet':
								no_1x1_df = ablation_df.query('Method in ["L1 No 1x1 pruning", "L2 No 1x1 pruning"]').T[dataset, flops]
								weights_only_df = ablation_df.query('Method in ["L1 1x1 = weights only", "L2 1x1 = weights only"]').T[dataset, flops]
								no_1x1_val_df = ablation_validation_df.query('Method in ["L1 No 1x1 pruning", "L2 No 1x1 pruning"]').T[dataset, flops]
								weights_only_val_df = ablation_validation_df.query('Method in ["L1 1x1 = weights only", "L2 1x1 = weights only"]').T[dataset, flops]

								# no_1x1_results = no_1x1_df[no_1x1_df.index != 'α = 1'].max().tolist()  # Best α on test data
								# weights_only_results = weights_only_df[weights_only_df.index != 'α = 1'].max().tolist()  # Best α on test data
								no_1x1_results = [no_1x1_df[no_1x1_df.index != 'α = 1'].loc[row, col] for col, row in no_1x1_val_df.idxmax().items()]  # Best α on validation data
								weights_only_results = [weights_only_df[weights_only_df.index != 'α = 1'].loc[row, col] for col, row in weights_only_val_df.idxmax().items()]  # Best α on validation data

								ablation_methods = methods[:2] + methods[4:]

								# Bar graph with best α, no 1x1 pruning, 1x1 weights only, and uniform/random/unpruned
								with Bar(f'{dataset} ({flops}) - Ablation', self._figures, groups=ablation_methods, n=4, ylabel=ylabel) as bar:
									for i, results in enumerate(zip(best_results[:2], best_results[2:4], no_1x1_results, weights_only_results)):
										for j, (result, colour, label) in enumerate(zip(results, (CMAP(0), CMAP(.25), CMAP(.5), CMAP(.75)), ("Classic", "LeGR", "No 1x1 pruning", "1x1 = weights only"))):
											bar.plot(result, i, j, label=label, color=colour)
									for i, result in enumerate(other_results):
										bar.plot(result, i + 2, label="Baseline", color=CMAP(1), width=4)

							# Heatmaps of correlations of α rankings
							corr = df.corr().dropna(axis=0, how='all').dropna(axis=1, how='all')
							labels = [METHODS[key] for key in corr.index]
							with Heatmap(figname, self._figures, labels=labels) as hmap:
								hmap.plot(corr)
							with Heatmap(f'{figname} (Norms)', self._figures, xlabels=("Same", "Different"), ylabels=labels) as hmap:
								hmap.plot([
									[corr['L1']['L1 LeGR'], np.mean((corr['L1']['L2'], corr['L1']['L2 LeGR']))],
									[corr['L2']['L2 LeGR'], np.mean((corr['L2']['L1'], corr['L2']['L1 LeGR']))],
									[corr['L1 LeGR']['L1'], np.mean((corr['L1 LeGR']['L2'], corr['L1 LeGR']['L2 LeGR']))],
									[corr['L2 LeGR']['L2'], np.mean((corr['L2 LeGR']['L1'], corr['L2 LeGR']['L1 LeGR']))]
								])
							with Heatmap(f'{figname} (Methods)', self._figures, xlabels=("Same", "Different"), ylabels=labels) as hmap:
								hmap.plot([
									[corr['L1']['L2'], np.mean((corr['L1']['L1 LeGR'], corr['L1']['L2 LeGR']))],
									[corr['L2']['L1'], np.mean((corr['L2']['L1 LeGR'], corr['L2']['L2 LeGR']))],
									[corr['L1 LeGR']['L2 LeGR'], np.mean((corr['L1 LeGR']['L1'], corr['L1 LeGR']['L2']))],
									[corr['L2 LeGR']['L1 LeGR'], np.mean((corr['L2 LeGR']['L1'], corr['L2 LeGR']['L2']))]
								])

	@classmethod
	def _write_latex_line(cls, f, row, block_max, model, flops, method, models, flopses, methods):
		if flops == flopses[0] and method == methods[0]:  # First line of model
			f.write(fr" \multirow{{{len(flopses) * len(methods)}}}{{*}}{{{model}}}")
		f.write(" &")
		if method == methods[0]:  # First line of specific FLOPs
			f.write(fr" \multirow{{{len(methods)}}}{{*}}{{{flops}}}".replace('%', '\%'))
		f.write(f" & {method} & ")
		row_max = row.max()
		if any(v != row_max for v in row):
			f.write(" & ".join(f"${cls._format_latex(v, row_max, block_max)}$" for v in row))
		else:
			f.write(fr"\multicolumn{{{len(row)}}}{{c}}{{{cls._format_latex(row[0], float('inf'), float('inf'))}}}")  # Multi-row
			# f.write(f" & & {cls._format_latex(row[0], float('inf'), float('inf'))} & &")  # Central column
		f.write(r" \\")
		if method == methods[-1]:  # Last line of specific FLOPs
			if flops != flopses[-1]:  # But not last line of model
				f.write(r"\cmidrule{2-8}")
			elif model != models[-1]:  # Also last line of dataset but not last line overall
				f.write(r"\midrule")
		f.write("\n")

	@classmethod
	def _format_latex(cls, value, row_max, block_max):
		if value >= block_max:
			return fr"\textcolor{{red}}{{{cls._format_latex(value, row_max, float('inf'))}}}"
		if value >= row_max:
			return fr"\mathbf{{{cls._format_latex(value, float('inf'), float('inf'))}}}"
		return f"{value:.2%}".replace('%', r'\%')

	##############################
	# End of main code           #
	##############################

	# Make this class callable with arguments too without cluttering the __call__ method
	def run(self, *args, **kw):
		# Update with varargs
		kw.update(dict(zip(self._varargs, map(Path, args))))

		# Update with passed parameters
		for arg, value in list(kw.items()):
			if arg in self:
				self[arg] = value
				del kw[arg]
		self.extra.update(kw)

		return self()

	#TEMPLATE: CLI arguments (don't need to touch this if you didn't add manual arguments)
	def process_command_line_options(self):
		# Use long description (or the first one if there's only 1)
		ap = argparse.ArgumentParser(description=(ensure_iterable(self._description, str) * 2)[1], formatter_class=_CustomArgFormatter)

		for attr, arg in self._automatic.items():
			if isinstance(arg, PathArg):
				arg.flags = attr
			else:
				name_flag = '--' + attr.replace("_", "-")
				if name_flag not in arg.flags:
					arg.flags = (name_flag, *arg.flags)
			arg.add_to_ap(ap)

		#TEMPLATE: Add your manual arguments here
		# ap.add_argument('-c', '--complicated-argument', nargs=2, help="this is a complicated argument")

		ap.add_argument('--gui', '--force-gui', dest='_force_gui', action='store_true', help="force GUI (while keeping existing arguments)")
		ap.add_argument('-e', '--extra', nargs='+', metavar=("KEY VALUE"), action=StoreDictPairsAction, help="any extra keyword-value argument pairs")
		ap.parse_args(namespace=self)

		if self._force_gui:
			return self.gui()
		return True

	def gui(self):
		gui = GUI(self)
		gui.mainloop()
		return gui.ok


class GUI(Tk):
	#TEMPLATE: GUI for arguments (don't need to touch this if you didn't add manual arguments)
	def __init__(self, argspace, *args, **kw):
		super().__init__(*args, **kw)
		self.args = argspace
		self.ok = False

		self.title(ensure_iterable(self.args._description, str)[0])
		self.frame = Frame(self)
		self.frame.pack(fill=BOTH, expand=YES)

		self.auto_vars = {}
		path_frame = Frame(self.frame)
		path_frame.grid_columnconfigure(1, weight=1)
		path_row = 0
		chk_frame = Frame(self.frame)
		rad_frame = Frame(self.frame)
		rad_frame.grid_columnconfigure(1, weight=1)
		rad_row = 0
		sld_frame = Frame(self.frame)
		sld_frame.grid_columnconfigure(1, weight=1)
		sld_row = 0
		spin_frame = Frame(self.frame)
		spin_row = 0
		clr_frame = Frame(self.frame)
		clr_row = 0
		for attr, arg in self.args._automatic.items():
			if isinstance(arg, PathArg):
				self.auto_vars[attr] = PathVar(value=self.args[attr])
				lbl = Label(path_frame, text=f"{arg.name if arg.name else attr.title().replace('_', ' ')}:")
				lbl.grid(row=path_row, column=0, sticky='w')
				txt = Entry(path_frame, width=60, textvariable=self.auto_vars[attr])
				txt.grid(row=path_row, column=1)
				command = partial(self._browse_file, self.auto_vars[attr], arg.exts) if arg.exts else partial(self._browse_dir, self.auto_vars[attr])
				btn = Button(path_frame, text="Browse", command=command)
				btn.grid(row=path_row, column=2)
				ToolTip(lbl, arg.help)
				ToolTip(txt, arg.help)
				ToolTip(btn, arg.help)
				path_row += 1

			elif isinstance(arg, ChoiceArg):
				var_type = IntVar if arg.type is int else DoubleVar if arg.type is float else StringVar
				self.auto_vars[attr] = var_type(value=self.args[attr])
				lbl = Label(rad_frame, text=f"{arg.name if arg.name else arg.help if len(arg.help) <= 20 else attr.title().replace('_', ' ')}:")
				lbl.grid(row=rad_row, column=0, sticky='w')
				if not arg.name and len(arg.help) > 20:
					ToolTip(lbl, arg.help)
				current_frame = Frame(rad_frame)
				for choice in arg.choices:
					rad = Radiobutton(current_frame, text=str(choice), variable=self.auto_vars[attr], value=choice)
					rad.pack(side=LEFT)
				current_frame.grid(row=rad_row, column=1, sticky='w')
				rad_row += 1

			elif isinstance(arg, BoolArg):
				self.auto_vars[attr] = BooleanVar(value=self.args[attr])
				text = arg.name if arg.name else arg.help if len(arg.help) <= 80 else attr.title().replace('_', ' ')
				chk = Checkbutton(chk_frame, text=text, variable=self.auto_vars[attr], anchor='w')
				if not arg.name and len(arg.help) > 80:
					ToolTip(chk, arg.help)
				chk.pack(fill=X, expand=YES)

			elif isinstance(arg, NumberArg):
				var_type = IntVar if arg.type is int else DoubleVar
				self.auto_vars[attr] = var_type(value=self.args[attr])
				text = f"{arg.name if arg.name else attr.title().replace('_', ' ')}:"
				if arg.gui_type.lower() == 'spinbox':
					lbl = Label(spin_frame, text=text)
					lbl.grid(row=spin_row, column=0, sticky='w')
					spin = Spinbox(spin_frame, from_=arg.min, to=arg.max, increment=arg.step, textvariable=self.auto_vars[attr])
					spin.grid(row=spin_row, column=1, sticky='w')
					spin_row += 1
				elif arg.gui_type.lower() in {'slider', 'scale'}:
					lbl = Label(sld_frame, text=text)
					lbl.grid(row=sld_row, column=0, sticky='w')
					sld = Scale(sld_frame, from_=arg.min, to=arg.max, resolution=arg.step, variable=self.auto_vars[attr], orient=HORIZONTAL)
					sld.grid(row=sld_row, column=1, sticky='we')
					sld_row += 1
				ToolTip(lbl, arg.help)

			elif isinstance(arg, ColourArg):
				self.auto_vars[attr] = [IntVar(value=self.args[attr][i]) for i in range(3)]
				lbl = Label(clr_frame, text=f"{arg.name if arg.name else arg.help if len(arg.help) <= 20 else attr.title().replace('_', ' ')}:")
				lbl.grid(row=clr_row, column=0, sticky='w')
				if not arg.name and len(arg.help) > 20:
					ToolTip(lbl, arg.help)
				for channel, var, col in zip(("Red", "Green", "Blue"), self.auto_vars[attr], range(1, 4)):
					spin = Spinbox(clr_frame, from_=0, to=255, textvariable=var)
					spin.grid(row=clr_row, column=col, sticky='w')
					ToolTip(spin, channel)
				clr_btn = Button(clr_frame, text="◑", command=partial(self._pick_colour, self.auto_vars[attr]))
				clr_btn.grid(row=clr_row, column=4)
				ToolTip(clr_btn, "Colour Picker")
				clr_row += 1

			else:
				raise ValueError("Unknown argument type in automatic GUI construction")

		path_frame.pack(fill=X, expand=YES)
		rad_frame.pack(fill=X, expand=YES)
		chk_frame.pack(fill=X, expand=YES)
		sld_frame.pack(fill=X, expand=YES)
		spin_frame.pack(fill=X, expand=YES)
		clr_frame.pack(fill=X, expand=YES)

		#TEMPLATE: Add manual arguments here (can add them to the above auto frames as well)

		self.extra_frame = ExtraFrame(self.frame)
		for key, value in self.args.extra.items():
			self.extra_frame.add_pair(key, str(value))
		self.extra_frame.pack(fill=X, expand=YES)

		ok_btn = Button(self.frame, text="OK", command=self.confirm)
		ok_btn.pack()
		ok_btn.focus()

	def _browse_dir(self, target_var):
		init_dir = target_var.get()
		if init_dir:
			while not init_dir.is_dir():
				init_dir = init_dir.parent

		new_entry = filedialog.askdirectory(parent=self, initialdir=init_dir)
		if new_entry:
			target_var.set(new_entry)

	def _browse_file(self, target_var, exts=None):
		init_dir = target_var.get().parent
		if init_dir:
			while not init_dir.is_dir():
				init_dir = init_dir.parent

		if exts:
			new_entry = filedialog.askopenfilename(parent=self, filetypes=exts, initialdir=init_dir)
		else:
			new_entry = filedialog.askopenfilename(parent=self, initialdir=init_dir)

		if new_entry:
			target_var.set(new_entry)

	def _pick_colour(self, vars):
		colour = askcolor(parent=self.frame, initialcolor=tuple(var.get() for var in vars))[0]
		if colour:
			for var, c in zip(vars, colour):
				var.set(c)

	#TEMPLATE: Reading arguments from GUI (don't need to touch this if you didn't add manual arguments)
	def confirm(self):
		for attr, var in self.auto_vars.items():
			try:
				self.args[attr] = tuple(v.get() for v in var)
			except TypeError:
				self.args[attr] = var.get()

		self.args.extra.clear()
		for kw in self.extra_frame.pairs:
			key, value = kw.key_txt.get(), kw.value_txt.get()
			if key:
				d = self.args if key in self.args else self.args.extra
				try:
					d[key] = literal_eval(value)
				except ValueError:
					d[key] = value

		self.ok = True
		self.destroy()


class PathVar(Variable):
	def __init__(self, *args, value=None, **kw):
		super().__init__(*args, value=Path(value) if value is not None else None, **kw)

	def get(self):
		return Path(v) if (v := super().get()) else None

	def set(self, value):
		return super().set(Path(value))


class ExtraFrame(Frame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)
		self.pairs = []

		self.key_lbl = Label(self, width=30, text="Key", anchor='w')
		self.value_lbl = Label(self, width=30, text="Value", anchor='w')

		self.add_btn = Button(self, text="+", command=self.add_pair)
		self.add_btn.grid()
		ToolTip(self.add_btn, "Add a new key-value pair")

	def add_pair(self, key="", value=""):
		pair_frame = KWFrame(self, pady=2, key=key, value=value)
		self.pairs.append(pair_frame)
		pair_frame.grid(row=len(self.pairs), columnspan=3, sticky='w')
		self.update_labels_and_button()

	def update_labels_and_button(self):
		if self.pairs:
			self.key_lbl.grid(column=0, row=0, sticky='w')
			self.value_lbl.grid(column=1, row=0, sticky='w')
		else:
			self.key_lbl.grid_remove()
			self.value_lbl.grid_remove()
		self.add_btn.grid(row=len(self.pairs) + 1)


class KWFrame(Frame):
	def __init__(self, *args, key="", value="", **kw):
		super().__init__(*args, **kw)

		self.key_txt = Entry(self, width=30)
		self.key_txt.insert(0, key)
		self.key_txt.grid(column=0, row=0, sticky='w')

		self.value_txt = Entry(self, width=30)
		self.value_txt.insert(0, value)
		self.value_txt.grid(column=1, row=0, sticky='w')

		remove_btn = Button(self, text="-", command=self.remove)
		remove_btn.grid(column=2, row=0)
		ToolTip(remove_btn, "Remove this key-value pair")

	def remove(self):
		i = self.master.pairs.index(self)
		del self.master.pairs[i]
		for pair in self.master.pairs[i:]:
			pair.grid(row=pair.grid_info()['row'] - 1)
		self.master.update_labels_and_button()
		self.destroy()


# Custom formatter that respects \n characters
class _CustomArgFormatter(argparse.RawTextHelpFormatter):
	def _split_lines(self, text, width):
		text = super()._split_lines(text, width)
		new_text = []

		# loop through all the lines to create the correct wrapping for each line segment.
		for line in text:
			if not line:
				# this would be a new line.
				new_text.append(line)
				continue

			# wrap the line's help segment which preserves new lines but ensures line lengths are honored
			new_text.extend(textwrap.wrap(line, width))

		return new_text

# Wrappers for automatic arguments
class _AutoArg:  # Can't use ABC because it breaks pickling
	def __init__(self, default, help, *flags, gui_name=None):
		self.default = default
		self.help = help
		self.flags = flags
		self.name = gui_name

	@abstractmethod
	def add_to_ap(self, ap):
		pass


class PathArg(_AutoArg):
	def __init__(self, default, help, *extensions, **kw):
		super().__init__(default, help, **kw)
		self.exts = extensions

	def add_to_ap(self, ap):
		ap.add_argument(self.flags, type=Path, nargs='?', default=self.default, help=self.help.lower())


class BoolArg(_AutoArg):
	def add_to_ap(self, ap):
		name = next((arg for arg in self.flags if arg[:2] == '--'), self.flags[0]).strip('-')
		help = self.help.lower()

		no_f = lambda arg: '--no-' + arg.strip('-') if arg[:2] == '--' else None
		str_to_bool = lambda s: s.lower() in {'true', 'yes', 't', 'y', '1'}

		group = ap.add_mutually_exclusive_group()
		group.add_argument(*self.flags, dest=name, nargs='?', default=self.default, const=True, type=str_to_bool, help=help)
		group.add_argument(*filter(None, map(no_f, self.flags)), dest=name, action='store_false', help="do not " + help)


class ChoiceArg(_AutoArg):
	def __init__(self, default, choices, help, *flags, choice_descriptions=(), type=None, **kw):
		super().__init__(default, help, *flags, **kw)
		self.choices = choices
		self.choice_descriptions = choice_descriptions
		self.type = type

		if self.type is None:
			try:
				self.type = int if all(int(x) == x for x in self.choices) else float
			except ValueError:
				if all(isinstance(x, str) for x in self.choices):
					self.type = str

	def add_to_ap(self, ap):
		help = self.help.lower()
		if self.choice_descriptions:
			longest = max(len(str(choice)) for choice in self.choices)
			help += "\n" + "\n".join(f"\t{choice:>{longest}}: {description}" for choice, description in zip(self.choices, self.choice_descriptions)) + "\n"
		ap.add_argument(*self.flags, type=self.type, default=self.default, choices=self.choices, help=help)


class NumberArg(_AutoArg):
	def __init__(self, default, range, help, *flags, type=None, gui_type='slider', **kw):
		super().__init__(default, help, *flags, **kw)
		self.type = type
		if self.type is None:
			self.type = int if all(int(x) == x for x in range) else float
		self.min, self.max = range[:2]
		self.step = range[2] if len(range) > 2 else 1 if self.type is int else .01
		self.range = self.min, self.max, self.step
		self.gui_type = gui_type

	def _type(self, x):
		x = self.type(x)
		if not self.min <= x <= self.max:
			raise argparse.ArgumentTypeError(f"{self.flags[0]} {x} should be between {self.min} and {self.max}")
		return x

	def add_to_ap(self, ap):
		ap.add_argument(*self.flags, type=self._type, default=self.default, help=self.help.lower() + f" [{self.min} <= x <= {self.max}]")


class ColourArg(_AutoArg):
	def _type(self, x):
		x = int(x)
		if not 0 <= x <= 255:
			raise argparse.ArgumentTypeError("Only 0-255 RGB values are supported in colour arguments")
		return x

	def add_to_ap(self, ap):
		ap.add_argument(*self.flags, type=self._type, nargs=3, metavar=("R", "G", "B"), default=self.default, help=self.help.lower())


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


if __name__ == '__main__':
	main = Main()

	# If CLI arguments, read them
	if len(sys.argv) > 1:
		if not main.process_command_line_options():
			# If CLI arguments weren't processed successfully, exit
			sys.exit(2)

	# Otherwise get them from a GUI
	else:
		if not main.gui():
			# If GUI was cancelled, exit
			sys.exit(1)

	main()

else:
	# Make module callable (python>=3.5)
	make_module_callable(__name__, Main().run)
