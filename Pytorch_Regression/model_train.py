# 모델 학습
# train_losses, train_r2s, test_r2s = train_model(model=model, IsLogTransformed=False, criterion=criterion, 
#                                                 X_train=X_train_transformed_tc.to(device), y_train=y_train_tc.to(device),
#                                                 X_test =X_test_transformed_tc.to(device),  y_test=y_test_tc.to(device),
#                                                 epochs=n_epochs)

# training parameters
n_epochs = 3000   # number of epochs to run
batch_size = 50  # size of each batch 

#criterion = nn.MSELoss()  
#criterion = nn.HuberLoss() ##nn.L1Loss()   

file_no = archt_file_name.split('.')[1].split('_')[-1] 

print(f'device: {device}')
train_losses, train_r2s, test_r2s, best_model = train_model_batch(model=model, IsLogTransformed=False, criterion=criterion, 
                                                X_train=X_train_ts.to(device), y_train=y_train_ts.to(device),
                                                X_test =X_test_ts.to(device),  y_test=y_test_ts.to(device),
                                                epochs=n_epochs, batch_size=batch_size, train_logger_file=f'train_logger_{file_no}.txt')
