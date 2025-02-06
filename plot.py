#!/usr/bin/env python
# coding: utf-8
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import exp_configs

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--problem', default="mushrooms")
args = parser.parse_args()

x_metric = "epoch"  # "time"
savedir_base = f"results/{args.problem}"
# rcv1
# filterby_list = [
#                  {"opt": {"name": "sgd", "lr": exp_configs.sgd_lr[args.problem]}},
#                  {"opt": {"name": "adam", "lr": exp_configs.adam_lr[args.problem]}},
#                  {"opt": {"name": "hdm_diag_scalar", "lr": 1.0, "beta_lr": 0.01, "normalize": 1}},
#                  {"opt": {"name": "hdm_scalar_scalar", "lr": 0.1, "beta_lr": 0.1, "normalize": 1}},
#                  ]
# ijcnn
filterby_list = [
                 {"opt": {"name": "sgd", "lr": exp_configs.sgd_lr[args.problem]}},
                 {"opt": {"name": "adam", "lr": exp_configs.adam_lr[args.problem]}},
                 {"opt": {"name": "hdm_diag_scalar", "lr": 1.0, "beta_lr": 0.01, "normalize": 0}},
                 {"opt": {"name": "hdm_diag_scalar", "lr": 1.0, "beta_lr": 0.01, "normalize": 1}},
                 {"opt": {"name": "hdm_scalar_scalar", "lr": 1.0, "beta_lr": 0.01, "normalize": 1}},
                 ]
new_legend_list = ["opt.name","opt.lr"]
map_legend_list = {
    "sgd": "SGD",
    "adam": "Adam",
    "hdm_diag_scalar|None": "HDM_DIAG_SCALAR",
}

y_metric_list_convex = ['train_loss', 'val_acc', 'train_epoch_time',]

# get experiments
rm = hr.ResultManager(savedir_base=savedir_base,
                      filterby_list=filterby_list,
                      verbose=0)

rm.exp_list = sorted(rm.exp_list, key=lambda d: d['opt']['name'])
rm.exp_list_all = sorted(rm.exp_list_all, key=lambda d: d['opt']['name'])
rm.exp_groups['all'] = sorted(rm.exp_groups['all'], key=lambda d: d['opt']['name'])
for d in rm.exp_list:
    print(hu.hash_dict(d))

# dashboard variables
legend_list = ['opt.name']
title_list = ['dataset', 'model']
y_metrics = ['train_loss', 'val_acc']

# launch dashboard
hj.get_dashboard(rm, vars(), wide_display=True)
y_metric_list = ['train_loss', 'val_acc', 'train_epoch_time']
num_normal_measures = len(y_metric_list)
additional_metric = ['val_acc']
y_metric_list = y_metric_list + additional_metric
x_lim = 200
xlim_list = [[0, x_lim] for i in range(len(y_metric_list))]
if "cifar100" in savedir_base:
    ylim_list = [None] * num_normal_measures + [[0.65, 0.78], [0, 500]]
elif "cifar10" in savedir_base:
    ylim_list = [None] * num_normal_measures + [[0.85, 0.95], [0, 500]]
    ylim_list[0] = [1e-06, 5]
elif "fashion" in savedir_base:
    ylim_list = [None] * num_normal_measures + [[0.875, 0.935], [0, 500]]
    ylim_list[0] = [1e-06, 5]
elif "mlp" in savedir_base:
    ylim_list = [None] * num_normal_measures + [[0.95, 0.99], [0, 500]]
    ylim_list[0] = [1e-06, 5]
elif "svhn" in savedir_base:
    ylim_list = [None] * num_normal_measures + [[0.925, 0.975], [0, 500]]
elif "enc" in savedir_base:
    y_metric_list = ['train_loss', 'train_metric', 'val_metric', 'train_epoch_time', 'backtracks', 'agv_step_size', "orig_step", 'grad_norm', 'd_norm']
    num_normal_measures = len(y_metric_list)
    ylim_list = [None] * num_normal_measures + [[10, 1000], [0, 1000], [0, 10]]
    y_metric_list = y_metric_list + ['train_metric', 'val_metric', 'backtracks']
    x_lim = 100
    xlim_list = [[0, x_lim] for i in range(len(y_metric_list))]
elif "xl" in savedir_base:
    y_metric_list = ['train_loss', 'train_metric', 'val_metric', 'train_epoch_time', 'backtracks', 'agv_step_size', "orig_step", 'grad_norm', 'd_norm']
    num_normal_measures = len(y_metric_list)
    ylim_list = [None] * num_normal_measures + [[1, 500], [0, 1000], [0, 100]]
    x_lim = 100
    y_metric_list = y_metric_list + ['train_metric', 'val_metric', 'backtracks']
    xlim_list = [[0, x_lim] for i in range(len(y_metric_list))]
elif "mushrooms" in savedir_base:
    # x_metric = "iter"
    y_metric_list = y_metric_list_convex
    ylim_list = [None] * len(y_metric_list_convex)
    ylim_list[-1] = [0.99, 1.0001]
    x_lim = 2000
    xlim_list = [[0, x_lim] for i in range(len(y_metric_list))]
    xlim_list[-1] = None
elif "rcv1" in savedir_base:
    # x_metric = "iter"
    y_metric_list = y_metric_list_convex
    ylim_list = [None] * len(y_metric_list_convex)
    ylim_list[-1] = [0.9, 0.98]
    xlim_list = [[0, 35] for i in range(len(y_metric_list))]
    xlim_list[-1] = None
elif "ijcnn" in savedir_base:
    # x_metric = "iter"
    y_metric_list = y_metric_list_convex
    ylim_list = [None] * len(y_metric_list_convex)
    ylim_list[-1] = [0.96, 0.98]
    xlim_list = [[0, 35] for i in range(len(y_metric_list))]
    xlim_list[-1] = None
elif "w8a" in savedir_base:
    # x_metric = "iter"
    y_metric_list = y_metric_list_convex
    ylim_list = [None] * len(y_metric_list_convex)
    ylim_list[-1] = [0.94, 0.98]
    xlim_list = [[0, 35] for i in range(len(y_metric_list))]
    xlim_list[-1] = None

# f_eval
# new_legend_list = ["opt.name"]
# map_legend_list = {
#     "sls_zhangNM_polyak": "PoNoS",
#     "a": "PoNoS_reset0",
# }
# y_metric_list = ["diff_backtracks", "avg_backtracks"]
# num_normal_measures = len(y_metric_list)
# additional_metric = ['avg_backtracks']
# y_metric_list = y_metric_list + additional_metric
# xlim_list = [[0, 20000], [0, 20000], [0, 100]]
# ylim_list = [None, None, None, None]


if x_metric == "time" or x_metric == "avg_time":
    xlim_list = [None for i in range(len(y_metric_list))]

map_title_list = [{"mnist  | mlp": "mnist | mlp",
                   "svhn   | wrn_10": "svhn | wrn",
                  "cifar10| densenet121": "cifar10 | densenet121",
                  "cifar100| densenet121_100": "cifar100 | densenet121",
                  "cifar10| resnet34": "cifar10 | resnet34",
                  "cifar100| resnet34_100": "cifar10 | resnet34",
                  "fashion| efficientnet-b1": "fashion | efficientnet-b1",
                  "wikitext2| transformer_encoder": "wikitext2 | transformer_encoder",
                   "ptb    | transformer_xl": "ptb | transformer_xl",
                   "mushrooms| logistic": "mushrooms | RBF kernel",
                   "rcv1   | logistic": "rcv1 | RBF kernel",
                   "w8a    | logistic": "w8a | RBF kernel",
                   "ijcnn  | logistic": "ijcnn | RBF kernel"}]


map_ylabel_list = [{"agv_step_size": "average step size",
                    "val_acc": "test accuracy",
                    "backtracks": "# backtracks",
                    "train_loss": "train loss",
                    "orig_step": "initial step size",
                    "grad_norm": "gradient norm",
                    "train_epoch_time": "runtime (s)",
                    "train_metric": "train perplexity",
                    "val_metric": "test perplexity",
                    "d_norm": "direction norm",
                    "all_grad_norm": "gradient norm",
                    "all_orig_step": "initial step size",
                    "all_step_size": "step size",
                    "smooth_loss": "train loss",
                    "diff_backtracks": "difference of backtracks",
                    "avg_backtracks": "average of backtracks",}]
map_xlabel_list = [{"time": "cumulative runtime (s)"}]

import os

for i, y_metric in enumerate(y_metric_list):
    img_dir = f"img/{args.problem}"
    os.makedirs(img_dir,exist_ok=True)
    if i < num_normal_measures:
        pp = PdfPages(f"{img_dir}/{y_metric}.pdf")
    else:
        pp = PdfPages(f"{img_dir}/{y_metric}_focus.pdf")
    
    print(y_metric)
    default_loc = {"loc":"lower right"}
    if y_metric in ["train_loss", "backtracks", "grad_norm", "d_norm", "train_metric", "val_metric", "smooth_loss", "n_backtr", "all_grad_norm", "avg_backtracks", "diff_backtracks"]:
        default_loc = {"loc": "upper right"}
    fig_list = rm.get_plot_all(y_metric_list=[y_metric],
                               x_metric=x_metric,
                               figsize=(12, 6),
                               title_list=['dataset', 'model'],
                               legend_list=new_legend_list,
                               log_metric_list=['train_loss', 'val_metric', 'train_metric', 'grad_norm', 'all_losses', "all_step_size", "all_orig_step",
                                                'all_grad_norm', 'avg_sharp',  'agv_step_size', "avg_nus_meas", 'orig_step', "d_norm", "smooth_loss"], #, 'avg_time'],
                               ylim_list=[[ylim_list[i]]],
                               xlim_list=[[xlim_list[i]]],
                               map_legend_list=map_legend_list,
                               avg_across="runs",
                               map_title_list=map_title_list,
                               # cmap=cmap,
                               map_ylabel_list=map_ylabel_list,
                               map_xlabel_list=map_xlabel_list,
                               legend_kwargs=default_loc,
                               title_fontsize=22,
                               y_fontsize=22,
                               x_fontsize=22,
                               legend_fontsize=20,
                               ytick_fontsize=16,
                               xtick_fontsize=16,
                               )
    fig = fig_list[0]
    fig.savefig(pp, format='pdf')
    pp.close()


