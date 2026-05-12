class Param():
    
    def __init__(self, args):
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            warmup_proportion (float): The warmup ratio for learning rate.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {
            # [新增/修改] 初始化粒球聚类模块所需的参数
            'purity_train': 0.95,  # 训练阶段纯度阈值
            'purity_get_ball': 0.95,  # 获取球阶段纯度阈值
            # [关键] 最小球内样本数
            # 如果是在 Batch 内跑，建议设小一点，比如 2 或 3
            # 如果是在整个数据集跑（Epoch开始前），建议设 5 或 10
            'min_ball_train': 5,
            'min_ball_get_ball': 2,

            'min_ball_select_ball': 1,  # 选择球时的最小样本数
            'purity_select_ball': 0.95,  # 选择球时的纯度
            'lambda_gb': 0.05,  # lambda_gb 是一个超参数，建议设为 0.1 或 0.01，防止刚开始球乱跑带偏模型
             'momentum':0.99,   #动量系数：0.99 代表非常平滑的更新
            # common parameters
            'eval_monitor': ['acc'], 
            'train_batch_size': 32,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8, 
            'num_train_epochs': 100, 
            # method parameters
            'warmup_proportion': 0.1, 
            'grad_clip': [-1.0], 
            'lr': [1e-5],   
            'learning_rate_method': 'decay',
            'weight_decay': [0.2], 
            'aligned_method': ['ctc'], 
            'shared_dim': [256], # a hyperparameter of MAG
            'eps': 1e-9,  # a hyperparameter of MAG
            # parameters of loss
            'loss': 'InfoNCE', 
            'temperature': [0.5], 
            # parameters of multimodal fusion
            'max_depth': [5], 
            'beta_shift': [0.006], # a hyperparameter of MAG
            'dropout_prob': [0.5], 
            'output_droupout_prob':[0.0],
            'extra_encoder': False

        }
        return hyper_parameters 