model_params = {}
model_params["FNN"] = {"batch_size" : 80, "hidden_size" : [64, 128, 256, 512], "output_size" : 4, "learning_rate" : 0.001, "num_epochs" : 60, "model_path" : '/content/drive/MyDrive/MTP_v2/Soumadip/Trained_Model/best_model_fnn.pth'}
model_params["RNN"] = {"batch_size" : 80, "input_size" : 4, "hidden_size" : 100, "output_size" : 4, "num_layers" : 3, "learning_rate" : 0.001, "num_epochs" : 150, "model_path" : '/content/drive/MyDrive/MTP_v2/Soumadip/Trained_Model/best_model_rnn.pth'}
