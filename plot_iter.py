import pickle
import os
import json
import pprint
import matplotlib.pyplot as plt

problem = 'rcv1'
result_folder = f'results/{problem}'
exp_ids = os.listdir(result_folder)

filterby_list = [
                 {"opt": {"name": "adam", "lr": 0.1}},
                 {"opt": {"name": "sgd", "lr": 0.1}},
                 {"opt": {"name": "hdm_diag_scalar", "lr": 1, "beta_lr":0.1}},
                 {"opt": {"name": "hdm_scalar_scalar", "lr": 0.1, "beta_lr":0.1}},
                ]

filterby_set = set()
for filter_item in filterby_list:
    filterby_set.add(frozenset(filter_item["opt"].items()))

for exp_id in exp_ids:
    if exp_id == 'deleted':
        continue 

    json_file = os.path.join(result_folder,exp_id,'exp_dict.json')
    pkl_file = os.path.join(result_folder,exp_id,'score_list.pkl')

    with open(json_file,'r') as f:
        config_data = json.load(f)
    with open(pkl_file,'rb') as f:
        score_data = pickle.load(f)
        # pprint.pprint(score_data)
    
    opt_name = config_data["opt"]["name"]
    lr = config_data["opt"]["lr"]
    opt_config = config_data["opt"]
    opt_tuple = frozenset(tuple(opt_config.items()))

    if opt_tuple in filterby_set:
    # if 1:
        loss_iter = []
        for epoch_data in score_data:
            # loss_iter.append(epoch_data['all_losses'])
            loss_iter += epoch_data['all_losses']
        
        plt.plot(loss_iter,label=f"{opt_name}_{lr}")
x_limit = 2000
plt.xlim(0,x_limit)
# plt.ylim(1e-12,1)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Train Loss (Log)')
plt.title(problem)
plt.legend()
plt.savefig(f"img/{problem}/train_loss.pdf")