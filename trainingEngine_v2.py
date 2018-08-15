import time, logging, torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
import setup_params
from helper_functions import compute_multilabel_AUCs,stats_calculator_binary
from customized_models_v2 import ModelCustomizer

class ModelTrainingAndTesting:
    def __init__(self, model_identifier, model_cache_dir,  number_of_classes, learning_rate = 0.001,  model_ft = None, criterion = None, optimizer = 'Adam'):

        '''
        NOTE:
        1. Specifying number_of_classes is vitally important because this class treats binary and multi-label case very differently.
        2. To do weigthed cross entropy, user needs to pass in a customized loss function.
        3. This class can be used for test purpose by passing the trained model and skip the training session. An example would be:
        training_engine = ModelTrainingAndTestingMultiLabel(model_identifier = identifier, model_cache_dir = model_cache_dir, model_ft =cached_m, number_of_classes = 14, optimizer = None)
        # skip training engine, report on testset directly.
        predictions_on_testset = training_engine.report_on_testset_ten_crop(test_dataloader = test_dataloader)
        '''

        self.number_of_classes = number_of_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_identifier = model_identifier
        self.model_cache_dir = model_cache_dir
        self.learning_rate = learning_rate

        if model_ft is None:
            model_customizer = ModelCustomizer(setup_params.MODEL_NAME)
            model_ft = model_customizer.network_modifier(self.number_of_classes)
            self.model_ft = model_ft.to(self.device)
        else:
            self.model_ft = model_ft.to(self.device)

        assert optimizer in ['SGD', 'RMS', 'Adam'] or optimizer is None, 'only support SGD, RMS, Adam.'

        if optimizer == 'Adam':
            self.optimizer_ft = torch.optim.Adam(self.model_ft.parameters(), lr = self.learning_rate)

        elif optimizer == 'RMS':
            self.optimizer_ft = torch.optim.RMSprop(self.model_ft.parameters(), lr = self.learning_rate)
        else:
            self.optimizer_ft = torch.optim.SGD(self.model_ft.parameters(), lr = self.learning_rate, momentum=0.9)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer_ft, mode = "min", patience = 2, verbose = True)

        if criterion is None and self.number_of_classes > 2:
            self.criterion = nn.BCELoss()

        elif criterion is None and self.number_of_classes == 2:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def _train_step(self, train_dataloader):

        self.model_ft.train()
        loss_cum = 0
        probs = []
        ground_truth = []

        for images, labels in train_dataloader:
            images = images.to(self.device)
            if self.number_of_classes == 2:
                labels = labels.to(self.device).long().squeeze_()
            else:
                labels = labels.to(self.device)

            self.optimizer_ft.zero_grad()

            with torch.set_grad_enabled(mode = True):

                outputs = self.model_ft(images)
                loss = self.criterion(outputs, labels)
                loss_cum = loss_cum + loss.item()
                probs.append(outputs)

                ground_truth.append(labels)
                loss.backward()
                self.optimizer_ft.step()

        lp = torch.cat(probs, 0).detach().cpu().numpy()
        gt = torch.cat(ground_truth, 0).detach().cpu().numpy()

        if self.number_of_classes == 2:
            lp = lp[:,1] #only succeses prob

        return [loss_cum, lp, gt]


    def _validation_step(self, validation_dataloader):

        self.model_ft.eval()
        loss_cum = 0
        probs = []
        ground_truth = []

        for images, labels in validation_dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer_ft.zero_grad()

            with torch.set_grad_enabled(mode = False):

                outputs = self.model_ft(images)
                loss = self.criterion(outputs, labels)

                loss_cum = loss_cum + loss.item()
                ground_truth.append(labels)
                probs.append(outputs)

        lp = torch.cat(probs, 0).detach().cpu().numpy()
        gt = torch.cat(ground_truth, 0).detach().cpu().numpy()

        if self.number_of_classes == 2:
            lp = lp[:,1] #only succeses prob
        return [loss_cum, lp, gt]


    def start_engine(self, number_of_epochs, train_dataloader, validation_dataloader):

        since = time.time()
        max_stats = 0.5 #ROC
        summary_train_overtime = []
        summary_val_overtime = []

        logging.info("-"*20 + 'Trainig Session Started'+ "-"*20)

        for epoch in range(number_of_epochs):
            since = time.time()

            loss_train, probs_train, ground_truth_train = self._train_step(train_dataloader)

            if self.number_of_classes >2:
                auc_list_training = compute_multilabel_AUCs(ground_truth_train, probs_train, self.number_of_classes)
                summary_train_overtime.append(auc_list_training)
                mean_auc_training = np.mean(auc_list_training)

                logging.info(f'the training auc for 14 classes is {auc_list_training}')
                logging.info(f'the average training auc for {self.number_of_classes} classes is {mean_auc_training}')

            else: #binary
                summary_train =  stats_calculator_binary([probs_train, ground_truth_train])
                summary_train_overtime.append(summary_train)

                logging.info('the training confusion matrix is {}'.format(summary_train['Confusion_Matrix']))
                logging.info('the training f1 score is {:.4f}'.format(summary_train['F1']))
                logging.info('the training accuracy is {:.4f}'.format(summary_train['Acc']))
                logging.info('the training ROC is {:.4f}'.format(summary_train['ROC']))
                logging.info('the training loss is {:.4f}'.format(loss_train/len(ground_truth_train)))

            loss_val, probs_val, ground_truth_val = self._validation_step(validation_dataloader)

            if self.number_of_classes > 2:
                auc_list_validation = compute_multilabel_AUCs(ground_truth_val, probs_val, self.number_of_classes)
                auc_val = np.mean(auc_list_validation)
                summary_val_overtime.append(auc_list_validation)

                logging.info(f'the validation auc for 14 classes is {auc_list_validation}')
                logging.info(f'the average validation auc for 14 classes is {auc_val}')

            else:
                summary_val =  stats_calculator_binary([probs_val, ground_truth_val])
                auc_val = summary_val['ROC']
                summary_val_overtime.append(summary_val)
                logging.info('the validation confusion matrix is {}'.format(summary_val['Confusion_Matrix']))
                logging.info('the validation f1 score is {:.4f}'.format(summary_val['F1']))
                logging.info('the validation accuracy is {:.4f}'.format(summary_val['Acc']))
                logging.info('the validation ROC is {:.4f}'.format(summary_val['ROC']))
                logging.info('the validation loss is {:.4f}'.format(loss_val/len(ground_truth_val)))


            if auc_val > max_stats:
                max_stats = auc_val
                logging.info(f'this is the best model with the best ROC-AUC score {max_stats}')
                name_cached_model = self.model_cache_dir + self.model_identifier + '.pth.tar'
                torch.save(self.model_ft, name_cached_model)
                logging.info(f'the model {name_cached_model} cached')

            self.scheduler.step(metrics = loss_val)

            time_elapsed = time.time() - since
            logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logging.info("-"*20 + f'finished epoch {epoch + 1}'+ "-"*20)
            logging.info('')

        return [summary_train_overtime, summary_val_overtime]

    def report_on_testset(self, test_dataloader, ten_crop = True):
        if ten_crop:
            ground_truth = torch.FloatTensor().to(self.device)
            pred_prob = torch.FloatTensor().to(self.device)

            for images, label in test_dataloader:
                label = label.to(self.device)
                ground_truth = torch.cat((ground_truth, label), 0)
                batch_size, n_crops, channel, height, width = images.size()
                input_var = images.view(-1, channel, height, width).to(self.device)

                with torch.set_grad_enabled(mode = False):
                    output = self.model_ft(input_var)
                    output_mean = output.view(batch_size, n_crops, -1).mean(1)
                    pred_prob = torch.cat((pred_prob, output_mean.data), 0)
        else:
             _, pred_prob, ground_truth = self._validation_step(test_dataloader)

        if self.number_of_classes > 2:

            AUROCs = compute_multilabel_AUCs(ground_truth, pred_prob, self.number_of_classes)
            AUROC_avg = np.array(AUROCs).mean()
            logging.info(f'the test auc for 14 classes is {AUROCs}')
            return AUROCs

        else:
            report_stats = stats_calculator_binary([pred_prob.ravel(), ground_truth.ravel()])

            roc_score = report_stats['ROC']
            acc = report_stats['Acc']
            conf_matrix = report_stats['Confusion_Matrix']
            f1 = report_stats['F1']

            logging.info(f'the test roc score is {roc_score:.4f}')
            logging.info(f'the test f1 score is {f1:.4f}')
            logging.info(f'the test accuarcy is {acc:.4f}')
            logging.info(f'the test confusion_matrix is {conf_matrix}')

            return report_stats
