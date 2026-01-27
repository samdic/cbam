from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBotCBAM_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBotCBAM_2d import get_umamba_bot_2d_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler    #Modif by Mariana
import torch    #Modif by Mariana

class nnUNetTrainerUMambaBotCBAM(nnUNetTrainer):

    # Modif by NR
    # - argument "unpack_dataset: bool = True" is no more required by nnUNetTrainer class constructor
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 800
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 800

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), 
                                      self.initial_lr, 
                                      weight_decay=self.weight_decay,
                                      eps=1e-4)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    #End of Modif by Mariana
   #modified by Maroua Zoubir 
def build_network_architecture(self,
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        plans_manager = self.plans_manager
        configuration_manager = self.configuration_manager
        dataset_json = self.dataset_json

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(
                plans_manager, dataset_json, configuration_manager,
                num_input_channels, deep_supervision=enable_deep_supervision
            )
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_bot_3d_from_plans(
                plans_manager, dataset_json, configuration_manager,
                num_input_channels, deep_supervision=enable_deep_supervision
            )
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaBotCBAM:", type(model).__name__)
        return model
