import torch
from torch.nn.parallel import DistributedDataParallel
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.build import build_detection_train_loader, build_detection_test_loader
from detectron2.engine.defaults import DefaultTrainer
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm

from .data import ContextDatasetMapper
from .eval import COCOEvaluatorWithAR


def get_trainer(cfg, resume=False, freeze=False):
    """Convenience method for getting a Trainer from a Detectron2 config."""
    trainer = COCOContextTrainer(cfg, freeze)
    trainer.resume_or_load(resume=resume)
    return trainer


class ContextTrainer(DefaultTrainer):
    """A trainer which uses a ContextDatasetMapper."""
    
    def __init__(self, cfg, freeze=False):
        """Same as DefaultTrainer but allows option to freeze non-context parameters."""
        TrainerBase.__init__(self) # skip DefaultTrainer init; logic modified below
        
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        
        if freeze:
            for n, p in model.named_parameters():
                if not "roi_heads.long_term" in n:
                    print("Freezing", n)
                    p.requires_grad = False
            
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=ContextDatasetMapper(cfg, is_train=True))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=ContextDatasetMapper(cfg, is_train=False))
    
    
class COCOContextTrainer(ContextTrainer):
    """A trainer which uses a COCOEvaluator and a ContextDatasetMapper."""
    def __init__(self, cfg, freeze=False):
        super(COCOContextTrainer, self).__init__(cfg, freeze)
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluatorWithAR(dataset_name, cfg, distributed=True)