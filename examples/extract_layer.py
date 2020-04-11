import os

from pytorch_hebbian.utils import extract_layer_from_state_dict

state_dict_path = os.path.join("models", "heb-20200408-193344_m_1000_acc=0.929.pth")
extract_layer_from_state_dict(state_dict_path=state_dict_path, layer='1')
