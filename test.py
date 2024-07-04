import settings
import torch
import pdb

def createModel(fileName, modelName):
    model= torch.load(fileName)
    setattr(settings, modelName, model)
    #settings.tf = load_Transformer(None)
    #settings.ae = torch.load('trained_models/TrackerAE10.pt')
    #settings.hand_model = torch.load("trained_models/hand_model.pt")
    #settings.ae5 = torch.load("trained_models/TrackerAE5.pt")
    #settings.cnn20 = torch.load("trained_models/TrackerCNN20.pt",map_location=torch.device("cpu"))

def test_model_vec(tracker_data,output, modelName):
    model=getattr(settings, modelName)
    with torch.no_grad():
        model.eval()
        data = torch.tensor(tracker_data.ref()).float()
        data = data.unsqueeze(0)

        model_output,_,_ = model(data[:,0,:],data[:,1,:],data[:,2,:],data[:,3,:],data[:,4,:],data[:,5,:],data[:,6,:],data[:,7,:],data[:,8,:],data[:,9,:])
        model_output = model_output.detach().numpy()
        output.ref()[:] = model_output
        return output

def test_model_mat(tracker_data,output, modelName):
    model=getattr(settings, modelName)
    assert(output.rows()>0)
    with torch.no_grad():
        model.eval()
        data = torch.tensor(tracker_data.ref()).float()
        data = data.unsqueeze(0)
        model_output,_,_ = model(data[:,0,:],data[:,1,:],data[:,2,:],data[:,3,:],data[:,4,:],data[:,5,:],data[:,6,:],data[:,7,:],data[:,8,:],data[:,9,:]) # 1d
        model_output = model_output.view(output.rows(),-1) # 2d
        model_output=model_output.detach().numpy()
        output.ref()[:,:]=model_output
        return output

def test_tracker_cnn(tracker_data,output):
    with torch.no_grad():
        settings.cnn20.eval()
        data = torch.tensor(Tonumpy(tracker_data)).float()
        data = data.unsqueeze(0)
        model_output = settings.cnn20(data)
        model_output = model_output.detach().numpy()
        output.ref()[:] = model_output
        return output

def test_HandModel(tracker_data,output):
    with torch.no_grad():
        settings.hand_model.eval()
        data = torch.tensor(Tonumpy(tracker_data)).float()
        data = data.unsqueeze(0)
        model_output = settings.hand_model(data)
        model_output = model_output.detach().numpy()
        output.ref()[:] = model_output
        return output

def test_Transformer(tracker_data, output):
    input = torch.Tensor(Tonumpy(tracker_data))
    input = input.reshape(1, 40, 90)
    settings.tf.eval()
    with torch.no_grad():
        model_output = settings.tf.forward(input)
        model_output = model_output.reshape(-1, 82)
        model_output = model_output.cpu().numpy()
        output.ref()[:] = model_output[:, :-4]
        return output