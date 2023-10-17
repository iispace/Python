import numpy as np 
from torch.utils.data import TensorDataset, DataLoader
from JJ_train_logger import write_model_architecture, write_train_history

def train_model_batch(model, IsLogTransformed, criterion, X_train, y_train, X_test, y_test, epochs, batch_size, train_logger_file):
    train_losses = []
    train_r2s = []
    test_r2s = []
    best_r2 = -np.inf
    best_model = {}

    lr = [param_group['lr'] for param_group in model.optimizer.param_groups][0]
    weight_decay = [param_group['weight_decay'] for param_group in model.optimizer.param_groups][0]
    dropout_p = model.dropout_p
    print(f'loss function: {criterion.__class__.__name__}, learning_rate: {lr}, weight_decay: {weight_decay}, dropout_p: {dropout_p}, batch_size: {batch_size}')
    # 파일에 모델 구조 쓰기
    write_model_architecture(train_logger_file, criterion, model)

    # Create a DataLoader for batch training 
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            # forward feed
            output_train = model(batch_X)
            y_train_cpu = batch_y.detach().cpu().numpy()
            output_train_cpu = output_train.detach().cpu().numpy()

            train_r2s.append(get_accuracy(y_train_cpu, output_train_cpu, IsLogTransformed=IsLogTransformed))
            if IsLogTransformed:
                loss = criterion(batch_y, np.e ** output_train)
            else:
                loss = criterion(batch_y, output_train)

            train_losses.append(loss.item())

            # clear out the gradients from the last step loss.backward()
            model.optimizer.zero_grad()

            # backward propagation: calculate gradients
            loss.backward() 

            # update the weights
            model.optimizer.step()
        
        # evaluate accuracy at the end of each epoch
        with torch.no_grad():
            output_test = model(X_test)
            y_test_cpu = y_test.detach().cpu().numpy()
            output_test_cpu = output_test.detach().cpu().numpy()
            test_r2 = get_accuracy(y_test_cpu, output_test_cpu, IsLogTransformed=IsLogTransformed)
            test_r2s.append(test_r2)
            
            if best_r2 < test_r2:
                best_r2 = test_r2
                best_model = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': model.optimizer.state_dict(), 'r2_score': test_r2, 'loss': loss}

        if (epoch+1) % 50 == 0 or epoch + 1 == epochs:
            train_r2 = sum(train_r2s)/len(train_r2s)
            test_r2 = sum(test_r2s)/len(test_r2s)
            line = f"Epoch {epoch+1:>5}/{epochs:>5}, Train Loss: {loss.item():18.4f}, Train r2_score: {train_r2:.4f}, Test r2_score: {test_r2:.4f}, difference between train-test: {abs(train_r2 - test_r2):.4f}"
            print(line)
            # 파일에 train history 기록
            write_train_history(train_logger_file, line)
                
    return train_losses, train_r2s, test_r2s, best_model
