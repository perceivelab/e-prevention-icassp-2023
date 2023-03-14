import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
from utils import utils
from torchmetrics import MeanMetric
from monai.data.meta_tensor import MetaTensor
from statsmodels.distributions.empirical_distribution import ECDF

class Trainer:

    ''' Class to train the classifier '''

    def __init__(self, net, class_weights, optim, track=1):

        # Store model
        self.net = net
        # Store optimizer
        self.optim = optim
        # Create Loss
        if track == 1:
            self.criterion_label = nn.CrossEntropyLoss(weight = class_weights)
        else:
            self.criterion = nn.MSELoss()
            self.target_metric = MeanMetric()

        # select track
        self.track = track

    def train(self, args):
        if self.track == 1:
            return self.train_track1(args)
        else:
            return self.train_track2(args)
    
    def eval(self, args, split):
        if self.track == 1:
            return self.eval_track1(args, split)
        else:
            return self.eval_track2(args, split)

    def forward_batch(self, inputs, labels, split):
        ''' 
        send a batch to net and backpropagate 
        '''
        # Set network mode
        if split == 'train':
            self.net.train()
            torch.set_grad_enabled(True)   
        else:
            self.net.eval()
            torch.set_grad_enabled(False)
            
        # foward pass
        out = self.net(inputs)
        
        if split=='train':
            predicted_labels_logits = out
        else: # voting in val and test
            predicted_labels_logits = out.mean(dim=0).unsqueeze(0)

        # compute loss label
        loss_labels = self.criterion_label(predicted_labels_logits, labels)
        loss = loss_labels
        
        # calculate label predicted and scores
        _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
        predicted_scores = predicted_labels_logits.data.clone().detach().cpu()

        if split == 'train':
            # zero the gradient
            self.optim.zero_grad()

            # backpropagate, gradient clipping and weigths update
            loss.backward()
            self.optim.step()
          
        # return metrics and predictions
        metrics = {}
        metrics['loss'] = loss.item()
        predicted = predicted_labels, predicted_scores

        return metrics, predicted

    def forward_batch_testing(self, inputs):
        ''' 
        send a batch to net and backpropagate 
        TODO: in origin net and scaler were passed as function params, here we use self.
        '''

        # Set network mode
        self.net.eval()
        torch.set_grad_enabled(False)
            
        # foward pass
        out = self.net(inputs)
        predicted_labels_logits = out.mean(dim=0).unsqueeze(0)

        # calculate label predicted and scores
        _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
        predicted_scores = predicted_labels_logits.data.clone().detach().cpu()
        
        predicted = predicted_labels, predicted_scores

        return predicted

    def train_track1(self, args):

        scheduler = args.scheduler
        saver = args.saver
        saver.save_configuration(dict(vars(args)))
        tot_predicted_labels_last = {split:{} for split in args.loaders}

        if (saver is not None) and (args.ckpt_every <= 0):
            max_validation_accuracy = 0
            max_test_accuracy = 0
            save_this_epoch = False

        saver.watch_model(self.net)

        for epoch in range(args.epochs):
        
            try:
                for split in args.loaders:
                    
                    data_loader = args.loaders[split]

                    epoch_metrics = {}
                    tot_true_labels = []
                    tot_predicted_labels = []
                    tot_predicted_scores = []

                    for batch in tqdm(data_loader, desc=f'{split}, {epoch}/{args.epochs}'):
                        
                        # get inputs and labels
                        inputs = batch['data']#.to(args.device)
                        labels = batch['label']

                        tot_true_labels.extend(labels.tolist())

                        # move to device
                        inputs = inputs.to(args.device)
                        labels = labels.to(args.device)
                        #print(inputs.device, labels.device)

                        # forward batch
                        metrics, (predicted_labels, predicted_scores) = self.forward_batch(inputs, labels, split)
                        
                        tot_predicted_labels.extend(predicted_labels.tolist())
                        tot_predicted_scores.extend(predicted_scores.tolist())
                        
                        for k, v in metrics.items():
                            epoch_metrics[k]= epoch_metrics[k] + [v] if k in epoch_metrics else [v]
                    
                    # Run scheduler
                    if args.enable_scheduler and split=="train":
                        scheduler.step(sum(epoch_metrics['loss'])/len(epoch_metrics['loss']))

                    # Print metrics
                    for metric_name, metric_values in epoch_metrics.items():
                        avg_metric_value = sum(metric_values)/len(metric_values)
                        if saver is not None:
                            saver.log_scalar(f"{split}/{metric_name}_epoch", avg_metric_value, epoch)
                    
                    if saver is not None:
                        # Accuracy classification
                        accuracy = sklearn.metrics.accuracy_score(tot_true_labels, tot_predicted_labels)
                        saver.add_scalar(f"{split}/acc", accuracy, epoch)
    
                        if (saver is not None) and (args.ckpt_every <= 0):
                            # save model at max val accuracy
                            if (split == "val") and (accuracy >= max_validation_accuracy):
                                max_validation_accuracy = accuracy
                                save_this_epoch = True
                            if (split == "test") and (accuracy >= max_test_accuracy):
                                max_test_accuracy = accuracy
                                save_this_epoch = True
                        
                        if split != 'train':
                            # Save max accuracy
                            saver.add_scalar(f"{split}/max_acc", max_validation_accuracy, epoch)
                        
                        if split == 'train':
                            saver.add_scalar("lr", self.optim.param_groups[0]['lr'], epoch)
                        
                        # Accuracy Balanced classification
                        accuracy_balanced = sklearn.metrics.balanced_accuracy_score(tot_true_labels, tot_predicted_labels)
                        saver.add_scalar(f"{split}/balanced_acc", accuracy_balanced, epoch)

                        # Precision
                        precision = sklearn.metrics.precision_score(tot_true_labels, tot_predicted_labels, average='macro', zero_division=0)
                        saver.add_scalar(f"{split}/precision", precision, epoch)
                        # Precision Balanced
                        balanced_precision = sklearn.metrics.precision_score(tot_true_labels, tot_predicted_labels, average='weighted', zero_division=0)
                        saver.add_scalar(f"{split}/precision_balanced", balanced_precision, epoch)

                        # Recall
                        recall = sklearn.metrics.recall_score(tot_true_labels, tot_predicted_labels, average='macro')
                        saver.add_scalar(f"{split}/recall", recall, epoch)

                        # Recall Balanced
                        balanced_recall = sklearn.metrics.recall_score(tot_true_labels, tot_predicted_labels, average='weighted')
                        saver.add_scalar(f"{split}/recall_balanced", balanced_recall, epoch)

                        # F1 Score
                        f1score = sklearn.metrics.f1_score(tot_true_labels, tot_predicted_labels, average='macro')
                        saver.add_scalar(f"{split}/f1", f1score, epoch)

                        # F1 Score Balanced
                        balanced_f1 = sklearn.metrics.f1_score(tot_true_labels, tot_predicted_labels, average='weighted')
                        saver.add_scalar(f"{split}/f1_balanced", balanced_f1, epoch)
                        
                        # Prediction Agreement Rate: concordanza valutazione stesso campione tra epoca corrente e precedente
                        predictionAgreementRate, tot_predicted_labels_last[split] = utils.calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last[split])
                        saver.add_scalar(f"{split}/prediction_agreement_rate", predictionAgreementRate, epoch)
                        
                        # Confusion Matrix
                        cm_image = utils.plot_confusion_matrix(tot_true_labels, tot_predicted_labels, list(range(self.net.num_classes)), title=f"{split} confusion matrix", show_values=False)
                        saver.add_images(f"{split}/confusion_matrix", cm_image, epoch)

                        # Log all the above metrics at once
                        saver.log()

                    # Save checkpoint
                    if saver is not None:
                        if args.ckpt_every > 0:
                            if (split ==  "train") and (epoch % args.ckpt_every == 0):
                                saver.save_model(self.net, args.experiment, epoch)
                        else: # args.ckpt_every <= 0
                            if save_this_epoch:
                                for filename in glob.glob(str(saver.ckpt_path/(args.experiment+"_best_"+split+"_*"))):
                                    os.remove(filename)
                                saver.save_model(self.net, args.experiment+"_best_"+split, epoch)
                            save_this_epoch = False
            
            except KeyboardInterrupt:
                print('Caught Keyboard Interrupt: exiting...')
                break

    def eval_track1(self, args, split='test'):

        data_loader = args.loaders[split]
        ids = []
        predictions = []
        scores = []

        if split == 'test':

            for batch in tqdm(data_loader, desc=f'{split}'):

                # get inputs and labels
                inputs = batch['data']#.to(args.device)
                #print(args.split_path)
                ids.extend(batch['sample_id'])

                # move to device
                inputs = inputs.to(args.device)

                # forward
                predicted_labels, predicted_scores = self.forward_batch_testing(inputs)
                predictions.extend(predicted_labels.tolist())
                scores.extend(predicted_scores.tolist())

            return {
                'day': ids,
                'user': predictions,
                'logits': scores
            }

        else:

            tot_true_labels = []
            tot_predicted_labels = []
            tot_predicted_scores = []

            for batch in tqdm(data_loader, desc=f'{split}'):
                        
                # get inputs and labels
                inputs = batch['data']#.to(args.device)
                labels = batch['label']#.to(args.device)

                tot_true_labels.extend(labels.tolist())

                # move to device
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                #print(inputs.device, labels.device)

                # forward batch
                metrics, (predicted_labels, predicted_scores) = self.forward_batch(inputs, labels, split)
                
                tot_predicted_labels.extend(predicted_labels.tolist())
                tot_predicted_scores.extend(predicted_scores.tolist())

            # Print metrics
            accuracy = sklearn.metrics.accuracy_score(tot_true_labels, tot_predicted_labels)
            return {
                'label': tot_true_labels,
                'user': tot_predicted_labels,
                'logits': tot_predicted_scores,
                'accuracy': accuracy
            }

    def forward_batch_track2(self, x, label, args):

        # Set network mode
        if args.split == 'train':  # Training Set
            self.net.train()
            torch.set_grad_enabled(True)
        else:
            self.net.eval()
            torch.set_grad_enabled(False)

        # Forward pass
        
        if args.split!='train':
            # batch dimension is fake
            x = x.squeeze(0)
        
        if args.enable_variational and self.net.training:
            x_rec, mu, logvar = self.net(x)
            loss = self.net.loss(x_rec, x, mu, logvar)
        else:
            # Forward
            x_rec = self.net(x)
            # Compute loss
            loss = self.criterion(x_rec, x)

        # Backward step and Update parameters
        if args.split == 'train':  # Training Set
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # Compute MAE
        mae = (x - x_rec).abs().mean()
        
        # Compute reconstruction distance
        if len(x_rec.size())==2:
            # we have B, F
            rec_dist = (x - x_rec).pow(2).sum(1).sqrt()
        else: # we have B, F, T
            rec_dist = (x - x_rec).pow(2).sum(2).sum(1).sqrt()

        avg_rec_dist = rec_dist.mean()
        median_rec_dist = rec_dist.median()

        # Initialize metrics
        rec_dist_metric = avg_rec_dist.item() if args.split == 'train' else median_rec_dist.item()
        metrics = {'loss': loss.item(),
                   'mae': mae.item(),
                   'rec_dist': rec_dist_metric,
                   'diff': (x-x_rec).detach().cpu()}

        return x_rec, metrics

    def train_track2(self, args):

        scheduler = args.scheduler
        saver = args.saver
        args.dist_thresholds = None
        print(f"Enable Variational: {args.enable_variational}")

        if saver is not None:
            saver.save_configuration(dict(vars(args)))
            saver.watch_model(self.net)
            max_score = 0
            save_this_epoch = False
            
        if (saver is not None) and (args.ckpt_every <= 0):
            save_this_epoch = False

        # Initialize output metrics
        result_metrics = {s: {} for s in ['train', 'val']}

        # Process each epoch
        for epoch in range(args.epochs):

            try:
                # Define stuff
                train_rec_dists = None
                dist_thresholds = None

                # Process each split
                for split in ['train', 'val']:

                    # Data Loader
                    data_loader = args.loaders[split]

                    # Epoch metrics
                    epoch_metrics = {}
                    # Train: Initialize training reconstruction distances
                    if split == 'train':  # Training Set
                        train_rec_dists = []
                    #    train_diff = []
                    elif split == 'val':  # Validation Set
                        tot_true_labels = []
                        tot_predicted_scores = []
                        diff = []
                        not_printed_normal = True
                        not_printed_anomaly = True

                    # Process each batch
                    for batch in tqdm(data_loader, desc=f'{split}, {epoch}/{args.epochs}'):
                        
                        # data
                        x = batch['data']#.to(args.device)
                        labels = batch['label'] # in train the will be -1

                        # Validation
                        if split == 'val':
                            tot_true_labels.append(labels.tolist()[0])

                        # Move to device
                        x = x.to(args.device)
                        labels = labels.to(args.device)

                        # Forwad Batch
                        args.split = split
                        x_rec, metrics = self.forward_batch_track2(x, labels, args)

                        # Check NaN and Infs
                        if torch.isinf(x).any():
                            raise FloatingPointError('Found inf values in input')
                        if torch.isnan(x_rec[0]).all():
                            if split == 'train':
                                #print(batch, x_rec)
                                raise FloatingPointError('Found NaN values')
                            else:
                                print('Warning: Found NaN values')

                        # Training
                        if split == 'train':
                            # Keep track of reconstruction distance
                            train_rec_dists.append(metrics['rec_dist'])
                        # Validation
                        elif split == 'val':
                            # Keep track of reconstruction distance
                            tot_predicted_scores.append(metrics['rec_dist'])
                            diff.append(metrics['diff'].pow(2).sum(2).sqrt().as_tensor())
                            # Save plot of a sample (images are added only if max score is improving)
                            if not tot_true_labels[-1] and not_printed_normal:
                                normal_reconstruction = utils.plot_signals(x[0][0].cpu(), x_rec[0].cpu())
                                not_printed_normal = False
                            if tot_true_labels[-1] and not_printed_anomaly:
                                anomaly_reconstruction = utils.plot_signals(x[0][0].cpu(), x_rec[0].cpu())
                                not_printed_anomaly = False

                        # Log metrics
                        for k, v in metrics.items():
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

                    # Run scheduler
                    if args.enable_scheduler and split=="train":
                        scheduler.step(sum(epoch_metrics['loss'])/len(epoch_metrics['loss']))

                    # End epoch
                    if split == 'train':
                        # set model in eval mode
                        self.net.eval()
                        torch.set_grad_enabled(False)
                        train_diff = []
                        # Disable gradient
                        with torch.no_grad():
                            for batch in args.loaders['train_distribution']:

                                # load batch data
                                sig = batch['data'].to(args.device)
                                # fed to the net and reconstruct
                                try:
                                    rec, _, _ = self.net(sig)
                                except Exception as e:
                                    rec = self.net(sig)

                                # append the difference to the diff vector
                                train_diff.append((sig-rec).detach().cpu())

                        # Compute train distribution
                        train_diff = torch.vstack(train_diff)
                        train_dist = train_diff.pow(2).sum(2).sqrt().as_tensor()
                        # Calculate CDS from trainSet distance
                        #print(train_dist.mean(), train_dist.size())
                        #print("Calculating CDF from training set...")
                        ecdf = []
                        for channel in range(train_dist.size(1)):
                            ecdf.append(ECDF(train_dist[:, channel]))
                        
                    else:
                        anomaly_scores = []
                        #print("Calculating PDF of valSet...")
                        # Calculate PDF from testSet distance
                        for dist in diff:
                            for channel in range(dist.size(1)):
                                dist[:,channel] = torch.Tensor(ecdf[channel](dist[:,channel]))

                            anomaly_scores.append(dist.mean(1).median())
                        tot_predicted_scores = torch.stack(anomaly_scores).tolist()

                    # Log epoch metrics
                    del epoch_metrics['diff']
                    for metric_name, metric_values in epoch_metrics.items():
                        # Compute epoch average
                        avg_metric_value = sum(metric_values)/len(metric_values)
                        if isinstance(avg_metric_value, MetaTensor):
                            print(metric_name, avg_metric_value.__dict__.keys())
                            avg_metric_value = avg_metric_value  
                        if saver is not None:
                            # Dump to saver
                            saver.add_scalar(f"{split}/{metric_name}_epoch", avg_metric_value, epoch)
                        # Add to output results
                        result_metrics[split][metric_name] = result_metrics[split][metric_name] + [avg_metric_value] if k in result_metrics[split] else [avg_metric_value]
 
                    if split == 'train':  # Training Set
                        # Get stats
                        min_rec_dist = min(train_rec_dists)
                        max_rec_dist = max(train_rec_dists)

                        # Log minimum distance
                        saver.add_scalar(f"{split}/min_rec_distance_epoch", min_rec_dist, epoch)
                        # Log maximum distance
                        saver.add_scalar(f"{split}/max_rec_distance_epoch", max_rec_dist, epoch)

                        # Log learning rate
                        saver.add_scalar("lr", self.optim.param_groups[0]['lr'], epoch)
                        
                    # Validation: compute TPR and FPR
                    elif split == 'val':
                        # Compute MAE for normal data and anomalies
                        normal_idx = torch.where(torch.tensor(tot_true_labels) == 0)[0]
                        anomaly_idx = torch.where(torch.tensor(tot_true_labels) == 1)[0]

                        tot_mae = torch.tensor(epoch_metrics['mae'])
                        tot_mse = torch.tensor(tot_predicted_scores)

                        mae_normal = tot_mae[normal_idx].mean()
                        mse_normal = tot_mse[normal_idx].mean()

                        mae_anomaly = tot_mae[anomaly_idx].mean()
                        mse_anomaly = tot_mse[anomaly_idx].mean()

                        saver.add_scalar(f"{split}/mae_normal", mae_normal, epoch)
                        saver.add_scalar(f"{split}/mse_normal", mse_normal, epoch)
                        saver.add_scalar(f"{split}/mae_anomaly", mae_anomaly, epoch)
                        saver.add_scalar(f"{split}/mse_anomaly", mse_anomaly, epoch)

                        try:
                            # Compute Precision Recall Curve
                            precision, recall, _ = sklearn.metrics.precision_recall_curve(tot_true_labels, tot_predicted_scores)
                            # Compute ROC Curve
                            fpr, tpr, _ = sklearn.metrics.roc_curve(tot_true_labels, tot_predicted_scores)

                            # Compute AUROC
                            auroc = sklearn.metrics.auc(fpr, tpr)
                            saver.add_scalar(f"{split}/auroc", auroc, epoch)

                            # Compute AUPRC
                            auprc1 = sklearn.metrics.auc(recall, precision)
                            auprc2 = sklearn.metrics.average_precision_score(tot_true_labels, tot_predicted_scores)
                            saver.add_scalar(f"{split}/auprc1", auprc1, epoch)
                            saver.add_scalar(f"{split}/auprc2", auprc2, epoch)

                            # Compute harmonic mean of AUROC and AUPRC
                            harmonic_mean1 = self.target_metric(torch.tensor([auroc, auprc1]))
                            harmonic_mean2 = self.target_metric(torch.tensor([auroc, auprc2]))
                            saver.add_scalar(f"{split}/harmonic_mean1", harmonic_mean1, epoch)
                            saver.add_scalar(f"{split}/harmonic_mean2", harmonic_mean2, epoch)

                            # save max score and update figures
                            if harmonic_mean1 >= max_score:
                                max_score = harmonic_mean1
                                save_this_epoch = True
                                # Plot images
                                curves = utils.plot_curves(fpr, tpr, recall, precision)
                                saver.add_images(f"{split}/roc_prc", curves, epoch)
                                saver.add_images(f"{split}/normal_reconstruction", normal_reconstruction, epoch)
                                saver.add_images(f"{split}/anomaly_reconstruction", anomaly_reconstruction, epoch)

                            saver.add_scalar(f"{split}/max_score", max_score, epoch)

                        except ValueError:
                            saver.add_scalar(f"{split}/auroc", 0., epoch)
                            saver.add_scalar(f"{split}/auprc1", 0., epoch)
                            saver.add_scalar(f"{split}/auprc2", 0., epoch)
                            saver.add_scalar(f"{split}/harmonic_mean1", 0., epoch)
                            saver.add_scalar(f"{split}/harmonic_mean2", 0., epoch)

                    # Log all the above metrics at once
                    saver.log()

                # Save checkpoint
                if saver is not None:
                    if args.ckpt_every > 0:
                        if (split ==  "train") and (epoch % args.ckpt_every == 0):
                            saver.save_model(self.net, args.experiment, epoch)
                    else: # args.ckpt_every <= 0
                        if save_this_epoch:
                            for filename in glob.glob(str(saver.ckpt_path/(args.experiment+"_best_"+split+"_*"))):
                                os.remove(filename)
                            saver.save_model(self.net, args.experiment+"_best_"+split, epoch)
                        save_this_epoch = False

            except KeyboardInterrupt:
                print('Caught keyboard interrupt: saving checkpoint...')
                self.saver.save_checkpoint(self.net, metrics, "Model", epoch)
                break

            except FloatingPointError as err:
                print(f'Error: {err}')
                break

        return self.net, result_metrics

    def eval_track2(self, args, split='test'):

        data_loader = args.loaders[split]
        ids = []
        user_ids = []

        # Set network mode
        self.net.eval()
        torch.set_grad_enabled(False)

        if split == 'test':

            tot_predicted_scores = []

            for batch in tqdm(data_loader, desc=f'{split}'):

                # get inputs
                x = batch['data']#.to(args.device)
                ids.extend(batch['sample_id'])
                user_ids.extend(batch['user_id'])

                # move to device
                x = x.to(args.device)

                # forward
                args.split = split
                _, metrics = self.forward_batch_track2(x, None, args)
                tot_predicted_scores.append(metrics['rec_dist'])

            return {
                'user': torch.tensor(user_ids).tolist(),
                'day': torch.tensor(ids).tolist(),
                'status': tot_predicted_scores,
            }
        
        else:

            tot_true_labels = []
            tot_predicted_scores = []

            for batch in tqdm(data_loader, desc=f'{split}'):
                        
                # get inputs and labels
                x = batch['data']#.to(args.device)
                labels = batch['label']#.to(args.device)
                
                user_ids.extend(batch['user_id'])
                tot_true_labels.extend(labels.tolist())

                # move to device
                x = x.to(args.device)
                labels = labels.to(args.device)

                # Forwad Batch
                args.split = split
                _, metrics = self.forward_batch_track2(x, labels, args)
                
                tot_predicted_scores.append(metrics['rec_dist'])

            # Save metric
            precision, recall, _ = sklearn.metrics.precision_recall_curve(tot_true_labels, tot_predicted_scores)
            # Compute ROC Curve
            fpr, tpr, _ = sklearn.metrics.roc_curve(tot_true_labels, tot_predicted_scores)
            # Compute AUROC
            auroc = sklearn.metrics.auc(fpr, tpr)
            # Compute AUPRC
            auprc = sklearn.metrics.auc(recall, precision)
            # Compute harmonic mean of AUROC and AUPRC
            final_score = self.target_metric(torch.tensor([auroc, auprc]))
            return {
                'user': torch.tensor(user_ids).tolist(),
                'label': tot_true_labels,
                'status': tot_predicted_scores,
                'score': final_score
            }