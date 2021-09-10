# Elememt_mapping_Quantification_by_Bayesian_based_U-net_model


## conda environments
conda create -n elements_mapping python=3.7

conda activate elements_mapping

pip install segmentation-models-pytorch

conda install -c menpo opencv

conda install matplotlib

conda install -c conda-forge albumentations

conda install -c menpo opencv

#########################################
The following issue will occur

Traceback (most recent call last):
  File "Model3_2.py", line 426, in <module>
    train_logs = train_epoch.run(train_loader)
  File "/home/xiaoyan/anaconda3/envs/elements_mapping/lib/python3.7/site-packages/segmentation_models_pytorch/utils/train.py", line 53, in run
    loss_logs = {self.loss.__name__: loss_meter.mean}
  File "/home/xiaoyan/anaconda3/envs/elements_mapping/lib/python3.7/site-packages/torch/nn/modules/module.py", line 781, in __getattr__
    type(self).__name__, name))
torch.nn.modules.module.ModuleAttributeError: 'JaccardLoss' object has no attribute '__name__'
  
This is a python issue not pytorch issue
  https://stackoverflow.com/questions/50542177/correct-handling-of-attributeerror-in-getattr-when-using-property
  https://github.com/pytorch/pytorch/issues/13981
  
Solution:
  Train.py:
  
  with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
    for x, y in iterator:
        x, y = x.to(self.device), y.to(self.device)
        loss, y_pred = self.batch_update(x, y)

        # update loss logs
        loss_value = loss.cpu().detach().numpy()
        loss_meter.add(loss_value)
        self.loss.__name__ = 'loss_name'
        loss_logs = {self.loss.__name__: loss_meter.mean}
        logs.update(loss_logs)
#########################################
  

https://user-images.githubusercontent.com/28584416/131885240-9179ab95-924f-47ca-a2ca-c4599e8deaaf.mp4

