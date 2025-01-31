## 1220 record
obs
- [ ] lr=1, fast in the initial period, slow later. final beta is near 0
- [ ] lr=0.1, slow in initial period, fast later. final beta is near 0.9
- [ ] gr_eps=1e-8, volatile in the initial period, high precision later
- [ ] gr_eps=1e-20, smooth in the initial period, low precision after

try
- [ ] lr=1, smaller beta
- [ ] gradually decreasing gr_eps

NOTICE
- [ ] for stochastic problem, tune problem on multiple seeds! rather, it overfit on the "best" param
- [ ] when using grid search, make sure the best parameter is contained in your grid
- [ ] for stochastic problem, train on large dataset, rather the random noise is too large
- [ ] tune one param a time, not grid search. determine a roadmap: tune learning rate first. then tune box contraints. and fix lr, tune other minor factors...
- [ ] set seeds carefully... i am tired of covered experiments!
- [ ] save your sweep and energy... i sincerely apologize for the power i wasted on runing logistic regression...
- [ ] be careful with grid search. do not contain all variable params in one search. the effect varies between all settings. there are too much noise and would disturb u from making a clear conclusion

tuning record!
need box?
- [ ] vowel 16: best is lr=0.1 beta_lr=0.01, than lr=0.1 beta_lr=1 (beta decreases fast)-> need a box? surprisingly, not getting better
- [ ] vowel 64: best is lr=0.1 beta_lr=0.1, than lr=1 (beta_lr=0.01,0.1), -> similarly, beta needs a box! -> not better
- [ ] vowel 128: best is lr=1 beta_lr=0.01>beta_lr=0.01, lr=10, beta_lr=0.01 -> the last needs a box
no need. it does no good.

need adaptive gr_eps?
- [ ] vowel 16: not distinguishable on (lr=0.1 beta_lr=0.01), but 
- [ ] vowel 64: 1e-8 is better, and there are spikes in 1e-20
- [ ] vowel 128:
i think this factor is minor. no one is significantly better than the other. 1e-20 typically results in a decrease in the final precision. Overall, 1e-8 is slightly stabler.

relax_eps? adaptive?
In all cases, 1.0 is the worst. and 1.0 initially performs well, but the steps of advantage differs.
In most casess. 1.5 is better on vowel dataset. no significant distinction between 1.5 and 2.5
this param is relatively important. more important than gr_eps, secondary to lr.

i tuned these parameters and draw these conclusions from mutliple seeds. but wheni go back to the all_optimizers test. i found that these params are even worse than the ones thatselected from single-seed grid search. what's wrong?

shut off wandb process
`ps aux | grep wandb`
`ps aux | grep wandb | grep -v grep | awk '{print $2}' | xargs kill -9`
clear wandb log
`rm -rf wandb/`


## 1224 record
- [x] add strong convexity (weight decay)
- [x] tricks on LBFGS: init, dampening, var reduction
- [x] implement dampening on param update, x-c*Pg
- [x] test dampening on osgm, and more dataset

## 1225 record
- [x] test dampening on osgm, and more dataset, not that effective 
- [ ] tricks of other optimizers, especially hd-like ones. maybe beta?
- [x] more test on logistic regression. add test for letter, satimage, segment
- [x] test dampening on other dataset
- [x] dampening is not that effective on middle dataset. because noise? No, just no help.
- [x] learning rate decay. No help
- [x] how to achieve initial fast rate? beta? NO, learning rate is the main factor
- [x] fix dampening update and test. not deterministic bette worser. almost no difference.
- [x] REtest dampening...on mnist
- [x] RErun all optimizers. no significant change.
- [x] the randomness is not fully controlled. fixed, train_test_split in sklearn
- [ ] new models: stochastic least square, overparameterized logistic regression
- [x] sgd_hd, how to write gradient? the same.
- [x] dampening: a heu for sgd with momentum

## 1227 record 
- [ ] overparameterized logistic regression
- [x] test dampening on weight decay 1e-4
- [x] gradient computation, using the same batch of data. but sgd_hd uses different data. now a better solution is available
- [x] use scalar for gradient precondition. effective on mnist dataset.
- [x] trade-off between scalar and matrix. relevant with the condition number of the problem? it is because the stepsize
- [x] use 1/k decaying stepsize.
- [ ] update infrequently
- [x] numerical stability constant: https://arxiv.org/pdf/2011.08181 
- [x] ablation of dampening on osmm2. not impacting osmm2
- [x] osmm2 on middle data. not the best. but not worse than osmm
- [x] osmm2 1/k stepsize on middle data. not that important.
- [x] osmm dampening effect on valid loss. not crutial
- [x] increase the regularizer of middle data. 1e-4 not exceed adam. 1e-2 even worse
- [x] compute the condition number of middle data
- [ ] mlp

## 1228 record
- [x] full batch run for middle data. nearly the best on deterministic settings

## 1229 record
- [x] implement overparameterized model
- [x] test overparameterized model on middle dataset 
- [x] AdamW on mlp+mnist
- [x] overparameterized model on satimage
- [ ] overparameterized model on segment

## 1230 record
currently, the experiments on mlp shows that Adam is not that competitive.
but this is not align with the results in the literature.
- [x] know the difference between Adam and AdamW: l2 vs weight decay
- [x] maybe i can write a blog discussing the development of optimizers and some important terms, like weight decay
- [ ] test overparameterized model with smaller batch size
- [ ] recover the experiments where Adam is superior
- [ ] why training a mlp is so slow?

Note:
1. Adam on mlp: 2 hidden layers with 1000. dropout used.
2. CNN + CIFAR10: Adam uses CNN, HD uses VGG

## 1231 record
- [x] any reference for overparameterized models? -> just large models, keep dataset unchanged.
- [x] test more models and dataset: CIFAR10 + vgg/resnet
- [x] search best parameters for CIFAR10: ongoing
- [ ] OSMM shows disadvantage w.r.t. wall time
- [ ] tune OSMM2 on mlp, plot the value of P

## 0101 record
- [x] warm up gpu before experiments, write a warmup config
- [x] remove next_iter for monotone oracle. test on MNIST128, almost no difference.
- [x] remove restart. 
- [x] increase the batch size of test loader
- [x] use CifarLoader: 7 min -> 3 min, but NaN (optimizer and len(train_loader)). use num_workers for acceleration.
- [x] use CifarLoader. control randomness? set seed at the beginning
- [x] add accuracy, loss scale?
- [x] osmm start from 0?
- [ ] add monotone oracle or not?
- [ ] PENDING: test different stepsize schedules

```python
'aug': {
        'flip': True,
        'translate': 2,
    }
test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'] or None)
```

monotone oracle
```python
next_data, next_target = None, None
next_data, next_target = next(iter(train_loader))
next_data, next_target = next_data.to(device), next_target.to(device)
def closure():
    next_output = model(next_data)
    loss = F.cross_entropy(next_output, next_target)
    return loss
```

restart
```python
restart = False
if not hasattr(config, 'stop_step'):
    config.stop_step = epochs * 2
if optimizer_name in ["OSMM","OSGM","OSMM2"] and epoch % config.stop_step == 0:
    restart = True
optimizer.step(closure, restart)
    restart = False
```

## 0115 record
- [ ] try to recover "linear convergence" for stochastic logistic
- [ ] use representative dataset
- [ ] what is the rate of SGD, stochastic NAG -> know interpolation and its effect on rate. also, stepsize
- [x] find interpolation experiments: https://github.com/IssamLaradji/ssn/tree/master

## 0116 record
- [ ] tricks: cosine lr for beta, and lr decay for P. also, add gradient norm for momentum

## 0119 record
- [ ] use the code in https://github.com/leonardogalli91/PoNoS, paper https://arxiv.org/pdf/2306.12747 