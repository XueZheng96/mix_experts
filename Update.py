#!/usr/bin/env python
# coding: utf-8

# In[1]:


class ClientUpdate(object):
    def __init__(self, train_set=None,  test_set=None, idx=None):
        
        self.loss_func = nn.NLLLoss()
        
        self.train_set, self.val_set = torch.utils.data.random_split(train_set, [800, 200], torch.Generator().manual_seed(idx))

        self.ldr_train = DataLoader(self.train_set, batch_size=10, shuffle=True)
        self.ldr_val = DataLoader(self.val_set, batch_size = 1, shuffle=True)

        self.test_set = test_set
        self.ldr_test = DataLoader(self.test_set, batch_size = 1, shuffle=True)
    
    def train(self, net, n_epochs,learning_rate):
        
        net.train()
        
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

        epoch_loss = []

        for iter in range(n_epochs):
            net.train()
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = (images, labels)
                net.zero_grad()
                log_probs = net(images.float())
    
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
#             val_acc, val_loss = self.validate(net,True)
#             print(val_acc)

        return net.state_dict(), epoch_loss[-1]
   
    
    def train_finetune(self, net, n_epochs, learning_rate):
        net.train()
        
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        
        patience = 10
        epoch_loss = []
        epoch_train_accuracy = []
        model_best = net.state_dict()
        train_acc_best = np.inf
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        
        for iter in range(n_epochs):
            net.train()
            batch_loss = []
            correct = 0
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = (images, labels)
                net.zero_grad()
                log_probs = net(images.float())
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                _, predicted = torch.max(log_probs.data, 1)
                correct += (predicted == labels).sum().item()
            train_accuracy = 100.00 * correct / len(self.ldr_train.dataset)
            epoch_train_accuracy.append(train_accuracy)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if(iter%5==0):
                val_acc, val_loss = self.validate(net,True)
                net.train()
                if(val_loss < val_loss_best - 0.01):
                    counter = 0
                    model_best = net.state_dict()
                    val_acc_best = val_acc
                    val_loss_best = val_loss
                    train_acc_best = train_accuracy
                    print("Iter: %d | %.2f" %(iter,val_acc_best))
                else:
                    counter = counter+1
                    
                # early stop
                if counter == patience:
                    return model_best, epoch_loss[-1], val_acc_best, train_acc_best
                    
    
        return model_best, epoch_loss[-1], val_acc_best, train_acc_best
     
        
    def train_mix(self, net_local, net_trans, gate, train_gate_only, n_epochs, early_stop, learning_rate, val):

        print("start train mix model")        
        gate.train()
        net_local.train()
        net_trans.train()

        if(train_gate_only):
            optimizer = torch.optim.Adam(gate.parameters(),lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(list(gate.parameters())+list(net_local.parameters()),lr=learning_rate)

      
        patience = 10
        epoch_loss = []
        gate_best = gate.state_dict()
        local_best = net_local.state_dict()
        trans_best = net_trans.state_dict()
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        gate_values_best = 0

        
        for iter in range(n_epochs):

            net_local.train()
            net_trans.train()
            gate.train()

            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = (images, labels)

                net_local.zero_grad()
                net_trans.zero_grad()
                gate.zero_grad()

                gate_weight = gate(images.float())

                # gate_weight*wi + gate_weight*fintuned
                local_prob = gate_weight*net_trans(images.float())+(1-gate_weight)*net_local(images.float())
                loss = self.loss_func(local_prob,labels)
  
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print(gate_weight)
            
            if(early_stop):
                if(iter%5==0):
                    val_acc, val_loss = self.validate_mix(net_local, net_trans, gate, True)
                    net_local.train()
                    net_trans.train()
                    gate.train()

                    if(val_loss < val_loss_best - 0.01):
                        counter = 0
                        gate_best = gate.state_dict()
                        local_best = net_local.state_dict()
                        trans_best = net_trans.state_dict()
                        val_acc_best = val_acc
                        val_loss_best = val_loss
                        #print("Iter: %d | %.2f" %(iter,val_acc_best))
                    else:
                        counter = counter + 1
                
                if counter == patience:
                    return gate_best, local_best, trans_best, epoch_loss[-1], val_acc_best


            return gate_best, local_best, trans_best, epoch_loss[-1], val_acc_best
    
    
    
    def validate(self,net,val):
        # if true validate dataset, if false use test detaset
        if(val):
            dataloader = self.ldr_val
        else:
            dataloader = self.ldr_test
       
        with torch.no_grad():
            net.eval()
            # validate
            val_loss = 0
            correct = 0
            for idx, (data, target) in enumerate(dataloader):
                data, target = (data, target)
                log_probs = net(data.float())

                val_loss += self.loss_func(log_probs, target).item()

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            val_loss /= len(dataloader.dataset)
            accuracy = 100.00 * correct / len(dataloader.dataset)
   
        return accuracy.item(), val_loss
    
    def validate_mix(self, net_l, net_t, gate, val):
        # if true validate dataset, if false use test detaset
        if(val):
            dataloader = self.ldr_val
        else:
            dataloader = self.ldr_test
        
        with torch.no_grad():
            net_l.eval()
            net_t.eval()
            gate.eval()
            val_loss = 0
            correct = 0
            gate_values = np.array([])
            label_values = np.array([])
            
            for idx, (data,target) in enumerate(dataloader):
                data, target = (data,target)
                gate_weight = gate(data.float())
                
                log_prob = gate_weight*net_t(data.float())+(1-gate_weight)*net_l(data.float())
                

                val_loss += self.loss_func(log_prob,target).item()
                y_pred = log_prob.data.max(1,keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        val_loss /= len(dataloader.dataset)
        accuracy = 100.00 * correct / len(dataloader.dataset)
        return accuracy.item(), val_loss
    
    

