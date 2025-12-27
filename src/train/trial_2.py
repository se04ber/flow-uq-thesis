import torch
import random
import copy
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np
import platform
import time
from model import AE_u_i, Hybrid_MD_RNN_AE_u_i
from model import RNN as RNN_new
from investigate_ensembles import RNN_old as RNN
from utils import get_RNN_loaders, get_Hybrid_loaders, mlready2dataset
from plotting import plotPredVsTargKVS, plot_flow_profile, plot_flow_profile_std


#Turn off for ensemble random weight initilisation diff
#torch.manual_seed(10)
#random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False


def train_RNN_u_i(loader_i, model_i, model_AE, optimizer_i, model_identifier_i, criterion, scaler, current_epoch):
    """The train_RNN_u_i_function trains the RNN model to be used with the
    single channel AE_u_i and computes the average loss on the training set.

    Args:
        loader_i:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model_i:
          Object of PyTorch Module class, i.e. the models to be trained.
        model_AE:
          Object of PyTorch Module class, i.e. the trained AE model.
        optimizer_i:
          The optimization algorithm applied during training.
        model_identifier_i:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch
          and model.
    """

    _epoch_loss = 0
    _counter = 0

    for _data, _targ in loader_i:
        _data = _data.float().to(device)
        _targ = _targ.float().to(device)

        with torch.cuda.amp.autocast():
            _pred_i = model_i(_data).to(device=device)
            n = int(torch.numel(_pred_i) / 256)
            _pred_i = torch.reshape(
                _pred_i, (n, 32, 2, 2, 2)).to(device=device)
            _targ_i = torch.reshape(
                _targ, (n, 32, 2, 2, 2)).to(device=device)
            _vel_pred = model_AE(_pred_i, y='get_MD_output').to(device=device)
            _vel_targ = model_AE(_targ_i, y='get_MD_output').to(device=device)
            _loss_i = criterion(_vel_pred, _vel_targ)
            _epoch_loss += _loss_i.item()
            _counter += 1

        _loss_i.backward(retain_graph=True)
        optimizer_i.step()
        optimizer_i.zero_grad()

    _avg_loss = _epoch_loss / _counter
    return _avg_loss


def valid_RNN_u_i(loader_i, model_i, model_AE, model_identifier_i, criterion, scaler, current_epoch):
    """The valid_RNN_u_i_ function validates the models and computes the average
    loss on the validation set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the models to be trained.
        optimizer:
          The optimization algorithm applied during training.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        model_identifier:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch
          and model.
    """

    _epoch_loss = 0
    _counter = 0

    for _data, _targ in loader_i:
        _data = _data.float().to(device)
        _targ = _targ.float().to(device)

        with torch.cuda.amp.autocast():
            _pred_i = model_i(_data).to(device=device)
            n = int(torch.numel(_pred_i) / 256)
            _pred_i = torch.reshape(
                _pred_i, (n, 32, 2, 2, 2)).to(device=device)
            _targ_i = torch.reshape(
                _targ, (n, 32, 2, 2, 2)).to(device=device)
            _vel_pred = model_AE(_pred_i, y='get_MD_output').to(device=device)
            _vel_targ = model_AE(_targ_i, y='get_MD_output').to(device=device)
            _loss_i = criterion(_vel_pred, _vel_targ)
            _epoch_loss += _loss_i.item()
            _counter += 1

    _avg_loss = _epoch_loss / _counter
    return _avg_loss


def trial_2_train_RNN_u_i():
    """The trial_2_train_RNN_single function trains an RNN model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        NONE

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    print('Starting Trial 2: Train RNN_u_i (KVS, MAE(velSpace), RNN_u_i)')

    print('Initializing RNN datasets.')
    _trains_i, _valids_i = get_RNN_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True)

    print('Initializing training parameters.')

    _criterion = nn.L1Loss()
    _file_prefix =    "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/'
    _file_prefix_AE = "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/'
    _alpha_string = '1e-4' #'1e-3' #'1e-4' #'1e-5'
    _alpha = 1e-4 #1e-3 #1e-4 #1e-5
    _num_layers = 1
    _seq_length = 25
    _model_identifier_RNN = f'RNN_LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}_i_Piet_all_ep150_actually'
    _model_identifier_AE = 'Model_AE_u_i_LR0_0001_i_Piet22' #'Model_AE_u_i_LR0_0001_i'
    print('Initializing RNN model.')

    _model_i = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)
    _optimizer_i = optim.Adam(_model_i.parameters(), lr=_alpha)
    _scaler = torch.cuda.amp.GradScaler()

    _model_AE = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_AE.load_state_dict(torch.load(
        f'{_file_prefix_AE}{_model_identifier_AE}', map_location='cpu'))
    _model_AE.eval()

    print('Beginning training.')

    for epoch in range(150):
        start = time.time()
        _avg_loss = 0
        for _train_i in _trains_i:
            loss = train_RNN_u_i(
                loader_i=_train_i,
                model_i=_model_i,
                model_AE=_model_AE,
                optimizer_i=_optimizer_i,
                model_identifier_i=_model_identifier_RNN,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch
            )
            _avg_loss += loss
        np.save("Loss_train_RNN.npy",_avg_loss)
        _avg_loss = _avg_loss/len(_trains_i)
        print('------------------------------------------------------------')
        print(f'Training Epoch: {epoch+1}')
        print(f'-> Avg loss {_avg_loss:.3f}')

        _avg_loss = 0
        end = time.time()
        print('Duration of one ML Calculation = 1 Coupling Cycle: ', end - start)

        for _valid_i in _valids_i:
            loss = valid_RNN_u_i(
                loader_i=_valid_i,
                model_i=_model_i,
                model_AE=_model_AE,
                model_identifier_i=_model_identifier_RNN,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch
            )
            _avg_loss += loss
        np.save("Loss_valid_RNN.npy",_avg_loss)
        _avg_loss = _avg_loss/len(_valids_i)
        print('------------------------------------------------------------')
        print(f'Validation Epoch: {epoch+1}')
        print(f'-> Avg loss {_avg_loss:.3f}')

    torch.save(
        _model_i.state_dict(),
        f'{_file_prefix}Model_{_model_identifier_RNN}'
    )



def uq_trial_2_train_RNN_u_i():
    """The trial_2_train_RNN_single function trains an RNN model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        NONE

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    print('Starting Trial 2: Train RNN_u_i (KVS, MAE(velSpace), RNN_u_i)')

    print('Initializing RNN datasets.')
    _trains_i, _valids_i = get_RNN_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True)

    print('Initializing training parameters.')

    _criterion = nn.L1Loss()
    _file_prefix =    "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/'
    _file_prefix_AE = "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/'
    
    #Settings for model i
    _alpha_strings = ['1e-4','1e-5','1e-3']
    _alphas         = [1e-4,1e-5,1e-3] #1e-5
    _num_layers_arr = [1,2,5]
    _seq_lengths    = [25,25,35]
    """
    rnn_settings=[]
    for i in range(len(_alphas)):
        for j in range(len(_num_layers_arr)):
            for k in range(len(_seq_lengths)):
                rnn_setting = {"_alpha_string":_alpha_strings[i],"alpha": _alphas[i], "_num_layers": _num_layers_arr[j], "_seq_length": _seq_lengths[k]}
                rnn_settings.append(rnn_setting)
    print(f"length settings rnn: {len(rnn_settings)}")
    """ 
    

    #Iterate over different randomized weights
    """
    for i in range(0,10,1):
        torch.manual_seed(i)
        random.seed(i)
    """
    #_model_identifier_RNN = f'RNN_LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}_{i}'#Iterate and train Rnns for all settings 
    """
    for i in range(len(rnn_settings)):
        _alphas = _alphas[i]
        _alphas_string = _alphas_strings[i]
        _num_layers = _num_layers_arr[i]
        _seq_length = _seq_lengths[i]
        _model_identifier_RNN = f'RNN_LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}_{i}'
        _model_identifier_AE = 'Model_AE_u_i_LR0_001_i'
        print('Initializing RNN model.')
    """
    
    #Iterate over different randomized weights
    i=0
    for j in range(0,10,1):
        torch.manual_seed(j)
        random.seed(j)
        _alpha = _alphas[i]
        _alpha_string = _alpha_strings[i]
        _num_layers = _num_layers_arr[i]
        _seq_length = _seq_lengths[i]
        _model_identifier_RNN = f'RNN_LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}_{i}_randomWeight{j}_ep15_newAE'
        _model_identifier_AE = "Model_AE_u_i_LRRandomWeightInit0_i_Piet_fullTraining_justX" #"Model_AE_u_i_LR0_0001_i_Piet22" #'Model_AE_u_i_LR0_001_i'
        print('Initializing RNN model.')

        _model_i = RNN(
            input_size=256,
            hidden_size=256,
            seq_size=_seq_length,
            num_layers=_num_layers,
            device=device
        ).to(device)
        _optimizer_i = optim.Adam(_model_i.parameters(), lr=_alpha)
        _scaler = torch.cuda.amp.GradScaler()
    
        _model_AE = AE_u_i(
            device=device,
            in_channels=1,
            out_channels=1,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        ).to(device)

        _model_AE.load_state_dict(torch.load(
            f'{_file_prefix_AE}{_model_identifier_AE}', map_location='cpu'))
        _model_AE.eval()

        print('Beginning training.')

        for epoch in range(15):
            start = time.time()
            _avg_loss = 0
            for _train_i in _trains_i:
                loss = train_RNN_u_i(
                    loader_i=_train_i,
                    model_i=_model_i,
                    model_AE=_model_AE,
                    optimizer_i=_optimizer_i,
                    model_identifier_i=_model_identifier_RNN,
                    criterion=_criterion,
                    scaler=_scaler,
                    current_epoch=epoch
                )
                _avg_loss += loss
            _avg_loss = _avg_loss/len(_trains_i)
            print('------------------------------------------------------------')
            print(f'Training Epoch: {epoch+1}')
            print(f'-> Avg loss {_avg_loss:.3f}')
            np.save("Loss_train_RNN_{i}.npy",_avg_loss)

            _avg_loss = 0
            end = time.time()
            print('Duration of one ML Calculation = 1 Coupling Cycle: ', end - start)

            for _valid_i in _valids_i:
                loss = valid_RNN_u_i(
                    loader_i=_valid_i,
                    model_i=_model_i,
                    model_AE=_model_AE,
                    model_identifier_i=_model_identifier_RNN,
                    criterion=_criterion,
                    scaler=_scaler,
                    current_epoch=epoch
                )
                _avg_loss += loss
            _avg_loss = _avg_loss/len(_valids_i)
        print('------------------------------------------------------------')
        print(f'Validation Epoch: {epoch+1}')
        print(f'-> Avg loss {_avg_loss:.3f}')

        torch.save(
            _model_i.state_dict(),
            f'{_file_prefix}Model_{_model_identifier_RNN}'
        )
        np.save("Loss_valid_RNN_{i}.npy",_avg_loss)




def prediction_retriever_hybrid(model_AE_directory, model_name_i, model_RNN_directory, model_name_RNN, dataset_name, save2file_name,t_max):
    """The prediction_retriever function is used to evaluate model performance
    of a trained model. This is done by loading the saved model, feeding it
    with datasets and then saving the corresponding predictions for later
    visual comparison.

    Args:
        model_directory:

        model_name:

        dataset_name:

    Returns:
        NONE
    """
    train_loaders, valid_loaders = get_Hybrid_loaders(
            data_distribution=dataset_name,
            batch_size=1,
            shuffle=False
        )
    print(len(train_loaders))

    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _num_layers = 1
    _seq_length = 25
    _model_RNN = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)

    _model_i.load_state_dict(torch.load(
        f'{model_AE_directory}{model_name_i}', map_location='cpu'))
    _model_i.eval()
    _model_RNN.load_state_dict(torch.load(
        f'{model_RNN_directory}{model_name_RNN}', map_location='cpu'))
    _model_RNN.eval()

    _model_Hybrid = Hybrid_MD_RNN_AE_u_i(
        device=device,
        AE_Model_x=_model_i,
        AE_Model_y=_model_i,
        AE_Model_z=_model_i,
        RNN_Model_x=_model_RNN,
        RNN_Model_y=_model_RNN,
        RNN_Model_z=_model_RNN,
        seq_length=_seq_length,
    ).to(device)

    _preds = torch.zeros(1, 3, 24, 24, 24).to(device=device)
    _targs = []

    for data, target in train_loaders[0]:
        data = data.float().to(device=device)
        data = torch.add(data, 1.0).float().to(device=device)
        # print('model_x(data) -> shape: ', data.shape)
        with torch.cuda.amp.autocast():
            _pred = _model_Hybrid(data)
            _preds = torch.cat((_preds, _pred), 0).to(device)
            _targs.append(target.cpu().detach().numpy())

    _preds = torch.add(_preds, -1.0).float().to(device=device)
    _preds = _preds[1:, :, :, :, :].cpu().detach().numpy()
    _targs = np.vstack(_targs)
    # _lbm = np.loadtxt('dataset_mlready/01_clean_lbm/kvs_20000_NE_lbm.csv', delimiter=";")
    # _lbm = _lbm.reshape(1000, 3)

    plotPredVsTargKVS(input_pred=_preds, input_targ=_targs, input_lbm=None,
            file_name=save2file_name,t_max=t_max)


def trial_2_Hybrid_verification(t_max=899):
    print('Starting Trial 2: Prediction Retriever (KVS + Aug, MAE, ReLU, AE_u_i, torch.add())')

    _model_AE_directory = ""#'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/'
    _model_RNN_directory = ""#'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/'
    _model_name_i = 'Model_AE_u_i_LR0_001_i'
    _model_name_RNN = 'Model_RNN_LR1e-5_Lay1_Seq25_i'
    _dataset_name = "get_KVS" #'get_Couette_eval'
    _save2file_name = "KVS_test" #'Couette_bottom_0_oscil_2_0_u_wall'

    prediction_retriever_hybrid(
        model_AE_directory=_model_AE_directory,
        model_name_i=_model_name_i,
        model_RNN_directory=_model_RNN_directory,
        model_name_RNN=_model_name_RNN,
        dataset_name=_dataset_name,
        save2file_name=_save2file_name,
        t_max=t_max
    )


def md_substitution_retriever(model_AE_directory, model_name_i, model_RNN_directory, model_name_RNN, id,t_max):
    """The prediction_retriever function is used to evaluate model performance
    of a trained model. This is done by loading the saved model, feeding it
    with datasets and then saving the corresponding predictions for later
    visual comparison.

    Args:
        model_directory:

        model_name:

        dataset_name:

    Returns:
        NONE
    """
    _directory = "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/KVS/Validation/'
    _id = id  # '22000_NW'
    _file_name = f"Data/Validation/processed_file_{_id}22.npy" #f'clean_kvs_combined_domain_init_{_id}.csv'
    _dataset = mlready2dataset(f'{_directory}{_file_name}')
    _dataset = _dataset[:, :, :,:,:]  #, 1:-1, 1:-1, 1:-1]
    print('Dataset shape: ', _dataset.shape)
    
    if(_dataset.shape == (900,1,24,24,24)):
        print(_dataset.shape)
        _dataset = np.concatenate([_dataset, _dataset, _dataset], axis=1)
        print(_dataset.shape)

    _targs = copy.deepcopy(_dataset[1:, :, :, :, :])
    _input_a = torch.from_numpy(copy.deepcopy(_dataset[:-1, :, :, :, :]))
    _input_b = torch.from_numpy(copy.deepcopy(_dataset[:-1, :, :, :, :]))

    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _num_layers = 1
    _seq_length = 25
    _model_RNN = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)

    _model_i.load_state_dict(torch.load(
        f'{model_AE_directory}{model_name_i}', map_location='cpu'))
    _model_i.eval()
    _model_RNN.load_state_dict(torch.load(
        f'{model_RNN_directory}{model_name_RNN}', map_location='cpu'))
    _model_RNN.eval()

    _model_Hybrid = Hybrid_MD_RNN_AE_u_i(
        device=device,
        AE_Model_x=_model_i,
        AE_Model_y=_model_i,
        AE_Model_z=_model_i,
        RNN_Model_x=_model_RNN,
        RNN_Model_y=_model_RNN,
        RNN_Model_z=_model_RNN,
        seq_length=_seq_length,
    ).to(device)

    '''
    [*_a] This portion creates a prediction dataset strictly on the basis of
    the MaMiCo input data
    '''
    _preds_a = torch.zeros(1, 3, 24, 24, 24).to(device=device)

    t = 0
    t_max = t_max

    while t < t_max:
        data = torch.reshape(_input_a[t, :, :, :, :], (1, 3, 24, 24, 24))
        data = torch.add(data, 1.0).float().to(device=device)
        with torch.cuda.amp.autocast():
            _pred = _model_Hybrid(data)
            _pred = torch.add(_pred, -1).float().to(device=device)
            _preds_a = torch.cat((_preds_a, _pred), 0).to(device)
        t += 1

    _preds_a = _preds_a[1:, :, :, :, :].cpu().detach().numpy()

    '''
    [*_b] This portion creates a prediction dataset on the basis of the recursive
    approach.
    '''
    _preds_b = torch.zeros(1, 3, 24, 24, 24).to(device=device)
    _input_b[:, 0, 3:21, 3:21, 3:21] = torch.zeros(
        t_max, 18, 18, 18) * _input_b[0, 0, 0, 0, 0]
    _input_b[:, 1, 3:21, 3:21, 3:21] = torch.zeros(
        t_max, 18, 18, 18) * _input_b[0, 1, 0, 0, 0]
    _input_b[:, 2, 3:21, 3:21, 3:21] = torch.zeros(
        t_max, 18, 18, 18) * _input_b[0, 2, 0, 0, 0]
    t_max = t_max
    t = 0
    while t < 25:
        data = torch.reshape(_input_b[t, :, :, :, :], (1, 3, 24, 24, 24))
        data = torch.add(data, 1.0).float().to(device=device)
        # print('model_x(data) -> shape: ', data.shape)
        with torch.cuda.amp.autocast():
            _pred = _model_Hybrid(data)
            _pred = torch.add(_pred, -1).float().to(device=device)
            _preds_b = torch.cat((_preds_b, _pred), 0).to(device)
        t += 1

    while t < t_max:
        data = torch.reshape(_input_b[t, :, :, :, :], (1, 3, 24, 24, 24))
        data[:, :, 3:22, 3:22, 3:22] = _preds_b[-1, :, 3:22, 3:22, 3:22]
        data = torch.add(data, 1.0).float().to(device=device)
        # print('model_x(data) -> shape: ', data.shape)
        with torch.cuda.amp.autocast():
            _pred = _model_Hybrid(data)
            _pred = torch.add(_pred, -1).float().to(device=device)
            _preds_b = torch.cat((_preds_b, _pred), 0).to(device)
        t += 1

    _preds_b = _preds_b[1:, :, :, :, :].cpu().detach().numpy()
    
    print("Gets here1")
    plot_flow_profile(
        np_datasets=[_targs, _preds_a, _preds_b],
        dataset_legends=[
            'MD', 'MD + Hybrid ML', 'Hybrid ML only'],
        save2file=f'{_id}_Fig_Maker_5_b_MD_vs_Hybrid_MD_vs_Hybrid_only_Piet_all_x',
        t_max=t_max
    )

    plot_flow_profile_std(
        np_datasets=[_targs, _preds_a, _preds_b],
        dataset_legends=[
            'MD', 'MD + Hybrid ML', 'Hybrid ML only'],
        save2file=f'{_id}_Fig_Maker_6_MD_vs_Hybrid_MD_vs_Hybrid_onlyPiet_all_x',
        t_max=t_max
    )


def fig_maker_3(id,t_max):
    print('Starting Trial 2: MD Substitution (KVS)')

    _model_AE_directory =  "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/'
    _model_RNN_directory = "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/'
    _model_name_i = "Model_AE_u_i_LRRandomWeightInit0_i_Piet_fullTraining_justX" #""Model_AE_u_i_LR0_0001_i_Piet22" #'Model_AE_u_i_LR0_0001_i' #'Model_AE_u_i_LR0_001_i'
    #_model_name_RNN = "Model_RNN_LR1e-4_Lay1_Seq25_i_Piet_all_OnlyX" #"Model_RNN_LR1e-4_Lay1_Seq25_i_Piet22_allComponents"  #
    _model_name_RNN  = "Model_RNN_old_LR1e-4_Lay1_Seq25_0_randomWeight0_ep15"
    #"Model_RNN_LR1e-4_Lay1_Seq25_i_Piet22_allComponents" #Bad #/instead _x" #_allComponents"  #
    #_model_name_RNN = "Model_RNN_LR1e-4_Lay1_Seq25_i" #Not so bad #"Model_RNN_LR1e-4_Lay1_Seq25_i_1459_allComponents" #"Model_RNN_LR1e-4_Lay1_Seq25_i" # 
    #_model_name_RNN = 'Model_RNN_LR1e-41e-5_Lay1_Seq25_i' # really bad 'Model_RNN_LR1e-41e-5_Lay1_Seq25_i' #'
    #_model_name_RNN = "Model_RNN_LR1e-4_Lay1_Seq25_0_randomWeight1" #ep15 and ep150(without) are both really bad 
    #_model_name_RNN ="Model_RNN_LR1e-5_Lay1_Seq25_i" ok but not doing turbulence good


    md_substitution_retriever(
        model_AE_directory=_model_AE_directory,
        model_name_i=_model_name_i,
        model_RNN_directory=_model_RNN_directory,
        model_name_RNN=_model_name_RNN,
        id=id,
        t_max=t_max
    )


if __name__ == "__main__":
    '''
    device = 'cpu'
    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _num_layers = 1
    _seq_length = 25
    _model_RNN = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)

    _model_i.eval()
    _model_RNN.eval()

    _model_Hybrid = Hybrid_MD_RNN_AE_u_i(
        device=device,
        AE_Model_x=_model_i,
        AE_Model_y=_model_i,
        AE_Model_z=_model_i,
        RNN_Model_x=_model_RNN,
        RNN_Model_y=_model_RNN,
        RNN_Model_z=_model_RNN,
        seq_length=_seq_length,
    ).to(device)

    print(platform.processor())

    _ins = torch.zeros((1, 24, 24, 24))
    start = time.time()
    _outs = _model_i(_ins)

    end = time.time()
    print('Duration of one ML Calculation = 1 Coupling Cycle: ', end - start)

    _ids = ['20000_NE', '22000_NW']  # , '26000_SE', '28000_SW']
    # _ids = ['22000_NW']
    for _id in _ids:
        fig_maker_3(id=_id)

    '''

    #trial_2_train_RNN_u_i()
    #uq_trial_2_train_RNN_u_i() #Ensemble

    fig_maker_3(id=1459,t_max=899)

    #trial_2_Hybrid_verification(t_max=899)
