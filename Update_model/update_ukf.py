import sys

from utils import *
from ukf import *

def main():
    overall_path = 'C:/Users/user/project/generate_ev_data/data/overall_pd.csv'
    trip_path = 'C:/Users/user/project/generate_ev_data/data/trip_pd.csv'
    
    model = ANN(layers=[3, 4, 8, 16, 8, 4, 1], activation='relu')
    # loss = torch.nn.functional.l1_loss
    try:
        checkpoint = torch.load('initial_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
    except:
        raise RuntimeError('No initial_model')
    
    v0_model = deepcopy(model)
    v_nums = [0]

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

        if l_1 > 0.05:
            # st = time.time()
            print(f'model update at route number {route_num}, ', end=' ')

            v0_model, ukf = train_ukf(model=v0_model, loader=train_loader,ukf_params=[1,2,0])
            print('Update complete')
            # print(time.time() - st)
            l_1 = 0
            for inp, tar in train_loader:
                output = v0_model(inp)
                v0_loss = loss(output, tar)
                l_1 += v0_loss.item()
            l_1 /= len(train_loader)