from detectron2.data.build import build_detection_train_loader, build_detection_test_loader
from detectron2.engine.defaults import DefaultTrainer

from .data import ContextDatasetMapper
from .eval import COCOEvaluatorWithAR


def get_trainer(cfg, resume=False):
    """
    Convenience method for getting a Trainer from a Detectron2
    config.
    """
    trainer = COCOContextTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    return trainer


class ContextTrainer(DefaultTrainer):
    """
    A trainer which uses a StoreFlipDatasetMapper.
    """
    def __init__(self, cfg):
        super(ContextTrainer, self).__init__(cfg)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=ContextDatasetMapper(cfg, is_train=True))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=ContextDatasetMapper(cfg, is_train=False))
    
    
class COCOContextTrainer(ContextTrainer):
    """
    A trainer which uses a COCOEvaluator and a ContextDatasetMapper.
    """
    def __init__(self, cfg):
        super(COCOContextTrainer, self).__init__(cfg)
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluatorWithAR(dataset_name, cfg, distributed=True)