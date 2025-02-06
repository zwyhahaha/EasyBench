from haven import haven_utils as hu

suffixes = ["", "_NM", "_trueNM", "_zhangNM", "_epochNM"]
armijo_list = ["sgd" + suff + "_armijo" for suff in suffixes]
sls_ada_list = ["sls_ada" + suff for suff in suffixes]
sls_polyak_list = ["sls" + suff + "_polyak" for suff in suffixes] + ["polyak"]
ours_opt_list = armijo_list + sls_polyak_list + sls_ada_list

hdm_opt_list = ["hdm_diag_scalar", "hdm_diag_diag", "hdm_scalar_scalar", "hdm_scalar_diag"]

exps = ["mnist_mlp","cifar10_resnet","cifar10_densenet","cifar100_res",
        "cifar100_dense","fashion_effb1","svhn_wrn","mushrooms","ijcnn","rcv1",
        "w8a","trans_enc","trans_xl"]

adam_lr = {
    "mnist_mlp": 1e-4,
    "cifar10_resnet": 1e-3,
    "cifar10_densenet": 1e-3,
    "cifar100_res": 1e-3,
    "cifar100_dense": 1e-3,
    "fashion_effb1": 1e-3,
    "svhn_wrn": 1e-3,
    "mushrooms": 10,
    "ijcnn": 0.1, # better than 1.0
    "rcv1": 0.1, # better than 1.0
    "w8a": 0.0001,
    "trans_enc": 2.5 * 1e-4,
    "trans_xl": 2.5 * 1e-4,
}

sgd_lr = {
    "mnist_mlp": 0.1,
    "cifar10_resnet": 0.1,
    "cifar10_densenet": 0.1,
    "cifar100_res": 0.1,
    "cifar100_dense": 0.1,
    "fashion_effb1": 0.1,
    "svhn_wrn": 0.1,
    "mushrooms": 1000,
    "ijcnn": 100,
    "rcv1": 100,
    "w8a": 0.001,
    "trans_enc": 0.5,
    "trans_xl": 0.25,
}

bench_opt = {}
for exp in exps:
    opt_configs = []
    opt_configs.append({"name": "sgd", "lr": sgd_lr[exp]})
    opt_configs.append({"name": "adam", "lr": adam_lr[exp]})
    bench_opt[exp] = opt_configs

hdm_scalar_scalar_lr = {
    "mnist_mlp": 1e-4,
    "rcv1": 0.1,
    "w8a": 0.00001, # beta = 0.1
}

hdm_diag_scalar_lr = {
    "mnist_mlp": 1e-4,
    "rcv1": 0.1,
    "w8a": 0.00001, # beta = 0.1
}

hdm_opt = {}
lr_lst = [1e-1,1e-2,1e-3,1e-4]
beta_lr_lst = [0.1,1.0,10.0]
lr_lst = [1.0]
beta_lr_lst = [0.01]
for exp in exps:
    opt_configs = []
    for lr in lr_lst:
        for beta_lr in beta_lr_lst:
            # opt_configs.append({"name": "hdm_diag_scalar", "lr": lr, "beta_lr":beta_lr, "normalize": 0})
            opt_configs.append({"name": "hdm_diag_scalar", "lr": lr, "beta_lr":beta_lr, "normalize": 1, "relax_coef": 1.1})
            # opt_configs.append({"name": "hdm_scalar_scalar", "lr": lr, "beta_lr":beta_lr, "normalize": 1, "relax_coef": 1.1})
    hdm_opt[exp] = opt_configs

long_run = 200
short_run = 75
many_runs = [0,1,2,3,4]
# Experiments definition
EXP_GROUPS = {
        "mnist_mlp":{"dataset":["mnist"],
            "model":["mlp"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": hdm_opt["mnist_mlp"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

        "cifar10_resnet":{"dataset":["cifar10"],
                "model":["resnet34"],
                "not_save_pth": True,
                "loss_func": ["softmax_loss"],
                "opt": hdm_opt["cifar10_resnet"],
                "acc_func":["softmax_accuracy"],
                "batch_size":[128],
                "max_epoch":[long_run],
                "runs":[0]},

        "cifar10_densenet":{"dataset":["cifar10"],
                "model":["densenet121"],
                "not_save_pth": True,
                "loss_func": ["softmax_loss"],
                "opt": hdm_opt["cifar10_densenet"],
                "acc_func":["softmax_accuracy"],
                "batch_size":[128],
                "max_epoch":[long_run],
                "runs":[0]},

        "cifar100_res":{"dataset":["cifar100"],
            "model":["resnet34_100"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": hdm_opt["cifar100_res"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

        "cifar100_dense":{"dataset":["cifar100"],
            "model":["densenet121_100"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": hdm_opt["cifar100_dense"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

        "fashion_effb1": {"dataset": ["fashion"],
                        "model": ["efficientnet-b1"],
                        "not_save_pth": True,
                        "loss_func": ["softmax_loss"],
                        "opt": hdm_opt["fashion_effb1"],
                        "acc_func": ["softmax_accuracy"],
                        "batch_size": [128],
                        "max_epoch":[long_run],
                        "runs":[0]},

        "svhn_wrn":{"dataset":["svhn"],
            "model":["wrn_10"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": hdm_opt["svhn_wrn"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[short_run],
            "runs":[0]},

        "mushrooms": {"dataset": ["mushrooms"],
                    "model": ["logistic"],
                    "loss_func": ['logistic_loss'],
                    "acc_func": ["logistic_accuracy"],
                    "opt": hdm_opt["mushrooms"],
                    "batch_size": [100],
                    "max_epoch": [10],
                    "runs": [0]},

        "ijcnn": {"dataset": ["ijcnn"],
                "model": ["logistic"],
                "loss_func": ['logistic_loss'],
                "acc_func": ["logistic_accuracy"],
                "opt": hdm_opt["ijcnn"],
                "batch_size": [100],
                "max_epoch": [35],
                "runs": [0]},

        "rcv1": {"dataset": ['rcv1'],
                    "model": ["logistic"],
                    "loss_func": ['logistic_loss'],
                    "acc_func": ["logistic_accuracy"],
                    "opt": hdm_opt["rcv1"],
                    "batch_size": [100],
                    "max_epoch": [35],
                    "runs": [0]},

        "w8a": {"dataset": ['w8a'],
                    "model": ["logistic"],
                    "loss_func": ['logistic_loss'],
                    "acc_func": ["logistic_accuracy"],
                    "opt": hdm_opt["w8a"],
                    "batch_size": [100],
                    "max_epoch": [35],
                    "runs": [0]},

        "trans_enc": {"dataset": ["wikitext2"],
                    "model": ["transformer_encoder"],
                    "not_save_pth": True,
                    "model_args": {"tgt_len": 35},
                    "loss_func": ["softmax_loss"],
                    "opt": bench_opt["trans_enc"],
                    "acc_func": ["ppl"],
                    "batch_size": [64],
                    "max_epoch": [100],
                    "runs": [0]},

        "trans_xl": {"dataset": ["ptb"],
                    "model": ["transformer_xl"],
                    "not_save_pth": True,
                    "model_args": {
                        "n_layer": 6,
                        "d_model": 512,
                        "n_head": 8,
                        "d_head": 64,
                        "d_inner": 2048,
                        "dropout": 0.1,
                        "dropatt": 0.0,
                        "tgt_len": 128,
                        "mem_len": 128,
                    },
                 "loss_func": ["softmax_loss"],
                  "opt": bench_opt["trans_xl"],
                 "acc_func": ["ppl"],
                 "batch_size": [64],
                 "max_epoch": [100],
                 "runs": [0]},

    #=========================================


            }

EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}
