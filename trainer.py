import torch
import random
from collections import defaultdict
from tqdm import tqdm
import pdb
import logging
import json
import numpy as np
import utils
from sklearn.metrics import f1_score



class Trainer():
    def __init__(self, model, criterion, optimizer, scheduler, log_path, weight_path, json_path, args, utils = utils):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.benchmark = True
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model = model

        self.criterion = criterion.to(self.device)

        self.log_path = log_path
        self.weight_path = weight_path
        self.json_path = json_path

        logging.basicConfig(filename=log_path)

        self.loss_dict = defaultdict(dict)

        self.loss_dict['loss']['train'] = []
        self.loss_dict['loss']['val'] = []
        self.loss_dict['loss']['test'] = []
        self.loss_dict['macro_f1']['train'] = []
        self.loss_dict['macro_f1']['val'] = []
        self.loss_dict['macro_f1']['test'] = []
        self.loss_dict['weighted_f1']['train'] = []
        self.loss_dict['weighted_f1']['val'] = []
        self.loss_dict['weighted_f1']['test'] = []
        self.loss_dict['acc']['train'] = []
        self.loss_dict['acc']['val'] = []
        self.loss_dict['acc']['test'] = []


        
        self.train_averagemeter = utils.AverageMeter()
        self.val_averagemeter = utils.AverageMeter()


        self.args = vars(args)

        self.inference_res = []
        self.inference_idx = []

       

        #save this so it easy for visualization in the future
    
    def fit(self, train_loader, val_loader, test_loader, epochs):
        for epoch in tqdm(range(epochs)):
            
            
            self.inference_res = []
            self.inference_idx = []
            
            #training! 
            train_loss, train_macro_f1, train_weighted_f1, train_acc = self.train(train_loader)

            #validating 
            val_loss,  val_macro_f1, val_weighted_f1, val_acc = self.validate(val_loader)
            test_loss, test_macro_f1, test_weighted_f1, test_acc = self.validate(test_loader)

            #update losses
            train_loss = round(train_loss, 4)
            val_loss = round(val_loss,4)
            test_loss = round(test_loss,4)

            self.loss_dict['loss']['train'].append(train_loss)
            self.loss_dict['loss']['val'].append(val_loss)
            self.loss_dict['loss']['test'].append(test_loss)

            self.loss_dict['macro_f1']['train'].append(train_macro_f1)
            self.loss_dict['macro_f1']['val'].append(val_macro_f1)
            self.loss_dict['macro_f1']['test'].append(test_macro_f1)


            self.loss_dict['weighted_f1']['train'].append(train_weighted_f1)
            self.loss_dict['weighted_f1']['val'].append(val_weighted_f1)
            self.loss_dict['weighted_f1']['test'].append(test_weighted_f1)


            self.loss_dict['acc']['train'].append(train_acc)
            self.loss_dict['acc']['val'].append(val_acc)
            self.loss_dict['acc']['test'].append(test_acc)

            loss_statement = "Model at Epoch: {}, train loss: {}, val loss: {}, test loss: {}".format(epoch, train_loss, val_loss, test_loss)
            macro_f1_statement = "Model at Epoch: {}, train macro_f1: {}, val macro_f1: {}, test macro_f1: {}".format(epoch, train_macro_f1, val_macro_f1, test_macro_f1)
            weighted_f1_statement = "Model at Epoch: {}, train weighted_f1: {}, val weighted_f1: {}, test weighted_f1: {}".format(epoch, train_weighted_f1, val_weighted_f1, test_weighted_f1)
            acc_statement = "Model at Epoch: {}, train acc: {}, val acc: {}, test acc: {}".format(epoch, train_acc, val_acc, test_acc)

            print(loss_statement)
            print('\n')
            print(macro_f1_statement)
            print('\n')
            print(weighted_f1_statement)
            print('\n')
            print(acc_statement)

            self.curr_val_metric = val_weighted_f1 + val_acc

            if epoch == 0:
                self.best_val_metric = self.curr_val_metric


                logging.warning(loss_statement)
                logging.warning(macro_f1_statement)
                logging.warning(weighted_f1_statement)
                logging.warning(acc_statement)

            else: 
                print(self.curr_val_metric)
                print(self.best_val_metric)
                if self.curr_val_metric > self.best_val_metric: 
                    print('UPDATE NEW SCORE')



                    #update loss
                    self.best_val_metric =  self.curr_val_metric

                    #save model weights
                    torch.save(self.model.state_dict(), self.weight_path)
                    
                    #log results
                    

                    logging.warning(loss_statement)
                    logging.warning(macro_f1_statement)
                    logging.warning(weighted_f1_statement)
                    logging.warning(acc_statement)
            
            self.scheduler.step()

                

        with open(self.json_path, "w") as outfile:
            json.dump(self.loss_dict, outfile)

        return self.loss_dict

    def train(self, loader):


        y_dict = {}
        y_dict['target'] = []
        y_dict['pred'] = [] 

        self.model.train()
        self.train_averagemeter.reset()
        for i, batch in enumerate(tqdm(loader)):
            
            
            labels = batch['eng'][:,:,-1].float().to(self.device)
            labels = self._roomreader_quantize_label_4class(labels)
            
            
            features = batch['s_openface'].float().to(self.device)


            #randomly shifting group order 

            randperm = torch.randperm(labels.shape[1])
            labels = labels[:,randperm]
            features = features[:, randperm, :,:]


            #different types of training 

            if self.args['train_level'] == 'individual':
                features = features.flatten(start_dim = 0, end_dim = 1)
                
                labels = labels.flatten(start_dim = 0, end_dim = 1)

            if self.args['personas']:
                personas = batch['personas'].flatten(start_dim = 0, end_dim = 1)
                out = self.model(features, personas)

            if self.args['contrastive']:
                out, other_loss = self.model(features)

            if self.args['video_feat']: 
                video_features = batch['video_feat'].float().to(self.device)
                video_features = video_features.flatten(start_dim = 0, end_dim = 1)
                out = self.model(features,video_features)

            if not self.args['video_feat']:
                out = self.model(features)
            

            if 'group' in self.args['train_level']:
                out = out.flatten(start_dim = 0, end_dim = 1)
                labels = labels.flatten(start_dim = 0, end_dim = 1)

            #get outputs and labels to compute f1
            y_dict['pred'].append(out)
            y_dict['target'].append(labels)

            
            loss = self._compute_loss(out, labels.long())
            
            if self.args['contrastive']:
                loss += other_loss
            

            self.train_averagemeter.update(loss.item())

            # if i == 0:
            #     print(out)
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
    
            loss.backward()

            # parameters update
            self.optimizer.step()

        
        preds = torch.concatenate(y_dict['pred'])
        targets = torch.concatenate(y_dict['target'])
        all_f1, macro_f1, weighted_f1,acc = self._compute_f1_acc(preds, targets)

        
           
        return self.train_averagemeter.avg, macro_f1,weighted_f1, acc

    def validate(self, loader):
        # put model in evaluation mode
        self.model.eval()
        self.val_averagemeter.reset()

        y_dict = {}
        y_dict['target'] = []
        y_dict['pred'] = [] 

        with torch.no_grad():
            for batch in loader:
                
                labels = batch['eng'][:,:,-1].float().to(self.device)

                labels = self._roomreader_quantize_label_4class(labels)
                features = batch['s_openface'].float().to(self.device)

                if self.args['train_level'] == 'individual':
                    features = features.flatten(start_dim = 0, end_dim = 1)
                    
                    labels = labels.flatten(start_dim = 0, end_dim = 1)

                if self.args['personas']:
                    personas = batch['personas'].flatten(start_dim = 0, end_dim = 1)
                    out = self.model(features, personas)

                if self.args['video_feat']:
                    video_features = batch['video_feat'].float().to(self.device)
                    video_features = video_features.flatten(start_dim = 0, end_dim = 1)
                    out = self.model(features,video_features)

                if self.args['contrastive']:
                    out, other_loss = self.model(features)

                if not self.args['video_feat']:
                    out = self.model(features)

                if 'group' in self.args['train_level']:
                    out = out.flatten(start_dim = 0, end_dim = 1)
                    labels = labels.flatten(start_dim = 0, end_dim = 1)
                
                y_dict['pred'].append(out)
                y_dict['target'].append(labels)

                #save inference results 
                self.inference_res.append(torch.max(out, dim = out.dim() - 1).indices)
                self.inference_idx.append(batch['index'])

                
                loss = self._compute_loss(out, labels.long())

                if self.args['contrastive']:
                    loss += other_loss

                self.val_averagemeter.update(loss.item())

        preds = torch.concatenate(y_dict['pred'])
        targets = torch.concatenate(y_dict['target'])
        all_f1, macro_f1, weighted_f1, acc = self._compute_f1_acc(preds, targets)

        print('\n')
        print(all_f1)

        
        return self.val_averagemeter.avg, macro_f1, weighted_f1, acc


    
    def _compute_loss(self, pred, target):
        loss = self.criterion(pred, target)

        return loss

    def _compute_f1_acc(self, pred, target):
        pred = pred.detach()
        pred = torch.max(pred, dim = pred.dim() - 1).indices
        print(pred)
        macro_f1 = f1_score(pred.cpu(), target.cpu(), average = 'macro')
        weighted_f1 = f1_score(pred.cpu(), target.cpu(), average = 'weighted')
        acc = (pred == target).float().mean()

        
        return f1_score(pred.cpu(), target.cpu(), average = None), round(macro_f1,4), round(weighted_f1,4), round(acc.item(),4) 

    
    def _roomreader_scale_label(self,target):
        return (target + 2)/4

    def _roomreader_quantize_label_4class(self,target):
        target = (target + 2)
        target = torch.clip(target, min = 0, max = 3)
        target = torch.floor(target)

        return target

    def _roomreader_quantize_label(self,target):
        #change to 9 level scale
        return torch.round(target * 2) + 4 




