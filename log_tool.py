from torch.utils.tensorboard import SummaryWriter
from MemoryDictionary import MemoryDictionary

def log_mem_dict(writer:SummaryWriter,mem_dict:MemoryDictionary,step:int) -> None:
    """
    writerはTensorboardに記録するための
    """