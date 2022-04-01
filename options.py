import os, shutil
from pyhocon import ConfigFactory

class DefaultOption(object):
    def __init__(self):
        # general option
        self.save_path = "./save/"
        self.data_path = "/dataset/"
        self.dataset = "cifar10"
        self.seed = 0
        self.nGPU = 1
        self.gpu = 0
        
        # dataloader option
        self.worker = 4

        # training option
        self.train = False

        # optimization option
        self.epcohs = 200
        self.batch_size = 128
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.optimizer = "SGD"

        self.lr = 0.01
        self.lr_policy = "multi_step"
        self.power = 1
        self.step = [0.6, 0.8]
        self.endlr = 0.0001
        self.lr_gamma = 0.1

        # model option
        self.model_name = "resnet18"
        self.load_state_dict = None
        

class Option(DefaultOption):
    def __init__(self, conf_path, args):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)
        self.save_path = self.conf['save_path']
        self.data_path = self.conf['data_path']
        self.dataset = self.conf['dataset']
        self.seed = getattr(self.conf, "seed", 0)
        self.nGPU = getattr(self.conf, "nGPU", 0)
        self.gpu = getattr(self.conf, "GPU", 0)
        
        # dataloader option
        self.worker = getattr(self.conf, "worker", 4)

        # training option
        self.train = self.conf['train']

        # optimization option
        self.epochs = self.conf['epochs']
        self.batch_size = self.conf['batch_size']
        self.weight_decay = self.conf['weight_decay']
        self.optimizer = self.conf['optimizer']

        if self.optimizer.lower() == "sgd":
            self.momentum = self.conf['momentum']
            self.nesterov = self.conf['nesterov']
        
        elif self.optimizer.lower() == "adam":
            self.adam_alpha = self.conf['adam_alpha']
            self.adam_beta = self.conf['adam_beta']

        self.warmup = getattr(self.conf, "warmup", 0)

        self.lr = getattr(self.conf, "lr", 0.01)
        self.scheduler = getattr(self.conf, "scheduler", None)
        self.ml_step = getattr(self.conf, "ml_step", None)
        self.lr_gamma = getattr(self.conf, "lr_gamma", None)
        
        # model option
        self.model_name = self.conf['model_name']
        self.load_state_dict = getattr(self.conf, "load_state_dict", False)
        self.log_override = getattr(self.conf, "log_override", False)

        # get param option

        self.activation_step = getattr(self.conf, 'activation_step', None)
        self.activation_index = getattr(self.conf, 'activation_index', None)

        # logger option

        self.get_weight_param = getattr(self.conf, 'get_weight_param', False)
        self.get_weight_grad_param = getattr(self.conf, 'get_weight_grad_param', False)

    def set_save_path(self):
        self.save_path = os.path.join(self.save_path, f"log_{self.dataset}_{self.model_name}_bs{self.batch_size}_ep{self.epochs}_seed_{self.seed}/")
        if os.path.exists(self.save_path) :
            print(f"{self.save_path} is exists")
            if self.log_override:
                shutil.rmtree(self.save_path)
            else:
                print(f"load log path {self.save_path}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def print_parameters(self):
        for key, value in sorted(self.conf.items()):
            print(f"{key} : {value}")
