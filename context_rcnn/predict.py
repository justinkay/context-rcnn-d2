import torch

from detectron2.engine.defaults import DefaultPredictor


class ContextPredictor(DefaultPredictor):
    
    def __call__(self, original_image, dataset_mapper, location):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            bank:

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            context_feats, num_valid_context_items = dataset_mapper._get_bank_feats(location, flipped=False)
            
            inputs = {"image": image, "height": height, "width": width, "context_feats": context_feats, "num_valid_context_items": num_valid_context_items}
            predictions = self.model([inputs])[0]
            return predictions