import numpy as np
import torch
from trainer.base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import compute_variances

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                      optimizer,
                      metric_ftns,
                      config,
                      train_set,
                      valid_set,
                      test_set):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.batch_size = config['data_loader']['batch_size']
        self.d_type = self.config['data_loader']['data']
        self.n_clusters = self.model.n_clusters
        self.input_dim = self.model.input_dim
        
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        
        self.train_n_batches = int(np.ceil(float(config["data_loader"]["n_train"]) / float(self.batch_size)))
        self.valid_n_batches = int(np.ceil(float(config["data_loader"]["n_valid"]) / float(self.batch_size)))
        self.test_n_batches = int(np.ceil(float(config["data_loader"]["n_test"]) / float(self.batch_size)))

        self.do_validation = self.valid_set is not None
        self.lr_scheduler = optimizer
        self.log_step = 4  # reduce this if you want more logs

        self.metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):        
        self.metrics.reset()                
        y0_outs, y1_outs, t_outs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        y_trgs, t_trgs, te_trgs = torch.tensor([]).to(self.device), torch.tensor([]).to(dtype=torch.int64).to(self.device), torch.tensor([]).to(self.device)
        assigned_clusters = torch.tensor([]).to(self.device)
        for idx in range(self.train_n_batches):
            self.model.train()
            x = torch.FloatTensor(
                    self.train_set['X'][idx*self.batch_size:(idx+1)*self.batch_size]
                ).to(self.device)
            t = torch.Tensor(
                    self.train_set['T'][idx*self.batch_size:(idx+1)*self.batch_size]                    
                ).to(self.device)            
            y = torch.Tensor(
                    self.train_set['Y'][idx*self.batch_size:(idx+1)*self.batch_size]                    
                ).to(self.device)
            te = torch.Tensor(
                    self.train_set['TE'][idx*self.batch_size:(idx+1)*self.batch_size]                    
                ).to(self.device)

            self.optimizer.zero_grad()
            loss, y0_pred, y1_pred, t_pred, clusters = self.model.predict(x, t, y)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.metrics.update('loss', loss.item())
            
            if idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(idx),
                    loss.item(),
                ))
                
            clusters = torch.argmax(clusters, dim=1)
            assigned_clusters = torch.cat([assigned_clusters, clusters], 0)
            
            y0_outs, y1_outs = torch.cat([y0_outs, y0_pred]), torch.cat([y1_outs, y1_pred])
            y_trgs, t_trgs = torch.cat([y_trgs, y]), torch.cat([t_trgs, t])
            te_trgs = torch.cat([te_trgs, te])
            
            with torch.no_grad():
                self.model.eval()
                self.model.update_centers(x)

        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(t_trgs, y_trgs, te_trgs, y0_outs, y1_outs))
                
        log = self.metrics.result() 
        within_var, across_var = compute_variances(te_trgs, assigned_clusters, self.n_clusters)
        clusters, counts = assigned_clusters.unique(return_counts=True, sorted=True)
        
        log.update({'clusters': clusters.tolist()})
        log.update({'counts': counts.tolist()})
        log.update({'within_var':within_var})
        log.update({'across_var':across_var})  
        
        if self.do_validation:
            val_log = self._infer(self.valid_set, self.valid_n_batches)
            log.update(**{'val_' + k: v for k, v in val_log.items()})      
        return log

    def _infer(self, data_set, n_batches):
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            y0_outs, y1_outs, t_outs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            y_trgs, t_trgs, te_trgs = torch.tensor([]).to(self.device), torch.tensor([]).to(dtype=torch.int64).to(self.device), torch.tensor([]).to(self.device)
            assigned_clusters = torch.tensor([]).to(self.device)
            for idx in range(n_batches):
                x = torch.FloatTensor(
                        data_set['X'][idx*self.batch_size:(idx+1)*self.batch_size]                    
                    ).to(self.device)   
                t = torch.Tensor(
                        data_set['T'][idx*self.batch_size:(idx+1)*self.batch_size]                    
                    ).to(self.device)            
                y = torch.Tensor(
                        data_set['Y'][idx*self.batch_size:(idx+1)*self.batch_size]                    
                    ).to(self.device)
                te = torch.Tensor(
                        data_set['TE'][idx*self.batch_size:(idx+1)*self.batch_size]                    
                    ).to(self.device)
                
                loss, y0_pred, y1_pred, t_pred, clusters = self.model.predict(x, t, y)
                self.metrics.update('loss', loss.item())
                
                clusters = torch.argmax(clusters, dim=1)
                assigned_clusters = torch.cat([assigned_clusters, clusters])
                y0_outs, y1_outs = torch.cat([y0_outs, y0_pred]), torch.cat([y1_outs, y1_pred])
                y_trgs, t_trgs = torch.cat([y_trgs, y]), torch.cat([t_trgs, t])
                te_trgs = torch.cat([te_trgs, te])
                
                
        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(t_trgs, y_trgs, te_trgs, y0_outs, y1_outs))
                
        log = self.metrics.result()
        
        within_var, across_var = compute_variances(te_trgs, assigned_clusters, self.n_clusters)

        clusters, counts = assigned_clusters.unique(return_counts=True, sorted=True)
        log.update({'clusters': clusters.tolist()})
        log.update({'counts': counts.tolist()})
        log.update({'across_var':across_var})
        log.update({'within_var':within_var})
            
        return log
        

    def _test_epoch(self):
        PATH = str(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(torch.load(PATH)['state_dict'])
        self.model.eval()
        
        log = {}        
        train_log = self._infer(self.train_set, self.train_n_batches)
        valid_log = self._infer(self.valid_set, self.valid_n_batches)
        test_log = self._infer(self.test_set, self.test_n_batches)
        log.update(**{'train_' + k: v for k, v in train_log.items()})
        log.update(**{'val_' + k: v for k, v in valid_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
                        
        self.logger.info('='*100)
        self.logger.info('Inference is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:20s}: {}'.format(str(key), value))  
        self.logger.info('='*100)

        return log
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.batch_size
        total = self.config['data_loader']['n_train']
        return base.format(current, total, 100.0 * current / total)
        
        
