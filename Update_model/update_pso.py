import matplotlib.pyplot as plt
import os
from time import strftime,gmtime

from utils import *
from pso import *

def main():
    overall_path = 'C:/Users/user/project/generate_ev_data/data/overall_pd.csv'
    trip_path = 'C:/Users/user/project/generate_ev_data/data/trip_pd.csv'

    model = ANN(layers=[3, 4, 8, 16, 8, 4, 1], activation='relu')
    try:
        checkpoint = torch.load('initial_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
    except:
        raise RuntimeError('No initial_model')
    
    epoch = 40
    pop = 400
    lb = -0.5
    ub = 0.5
    w = 0.75
    c = [1, 1, 0.5]

    calibration_threshold = 0.05
    terminate_threshold = 0.01
    p_interval = 1

    log_dir = 'runs/pso/' + strftime("%m%d%H%M", gmtime()) + '/'

    v0_model = deepcopy(model)
    v_nums = [0]

    dict_keys = ['route_num', 'initial_model_loss', 'update_model_loss', 'update_trigger']
    dict_values = [[], [], [], []]
    loss_hist = dict(zip(dict_keys, dict_values))
    model_parameters = []
    model_parameters.append(model_parameter_list(v0_model))

    v0_model.eval()
    model.eval()
    for route_num in range(1, 100):
        routes = [route_num]
        ev_dataset = EV_dataset(overall_path, trip_path, v_nums, routes)
        train_loader = DataLoader(ev_dataset, batch_size=512, shuffle=False)

        l_0, l_1 = 0, 0
        for inp, tar in train_loader:
            output_ = model(inp)
            i_loss = loss(output_, tar)
            l_0 += i_loss.item()

            output_0 = v0_model(inp)
            v0_loss = loss(output_0, tar)
            l_1 += v0_loss.item()
        l_0 /= len(train_loader)
        l_1 /= len(train_loader)

        if l_1 > calibration_threshold:
            # st = time.time()
            print(f'model update at route number {route_num},', end=' ')
            # print(f'loss before update : {l_1}')
            loss_hist['update_trigger'].append(route_num)

            v0_model, pso = train_pso(model = v0_model, epoch=epoch, pop=pop, 
                                    loader = train_loader, log_dir= log_dir + str(route_num),
                                    terminate_threshold=terminate_threshold,
                                    p_interval=p_interval, w=w, lb=lb, ub=ub, c=c)
            
            # print(time.time() - st)
            l_1 = 0
            for inp, tar in train_loader:
                output = v0_model(inp)
                v0_loss = loss(output, tar)
                l_1 += v0_loss.item()
            l_1 /= len(train_loader)

            # print(f'loss after update : {l_1}')
            model_parameters.append(model_parameter_list(v0_model))
            print('Update complete')

        loss_hist['route_num'].append(route_num)
        loss_hist['initial_model_loss'].append(l_0)
        loss_hist['update_model_loss'].append(l_1)

    print('model updated {} times after initial model'.format(len(loss_hist['update_trigger'])))
    loss_hist['model_parameters'] = model_parameters
    np.save('./result/pso.npy', loss_hist)

    plt.plot(loss_hist['route_num'], loss_hist['initial_model_loss'], label = 'initial model', c='b')
    plt.plot(loss_hist['route_num'], loss_hist['update_model_loss'], label = 'updated model', c='g')
    plt.scatter(loss_hist['update_trigger'], [0 for _ in loss_hist['update_trigger']], label = 'calibration trigger', c='r')
    plt.legend()
    plt.xlabel('route number')
    plt.ylabel('MAE loss')
    plt.savefig('retrain_loss_pso.png')
    plt.show()

if __name__ == '__main__':
    main()