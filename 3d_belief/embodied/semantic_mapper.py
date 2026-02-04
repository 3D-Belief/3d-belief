import torch
import yaml
from typing import Callable, List, Dict
import torch.nn.functional as F
from utils.vision_utils import viz_feat, semantic_to_color
import matplotlib.pyplot as plt
import numpy as np

class SemanticMapper(torch.nn.Module):
    """
    Maps feature maps [b, c, h, w] to a one-channel semantic map [b, h, w] of integer labels.

    Modes:
    - 'one_hot': features are one-hot across the channel dimension.
    - 'embed': features are embeddings; use a text encoder to get label embeddings.
    """
    def __init__(
        self,
        config_path: str,
        mode: str = 'one_hot',
        text_encoder: Callable[[List[str]], torch.Tensor] = None,
        text_tokenizer: Callable[[List[str]], List[str]] = None,
        semantic_viz: str = 'seg',
    ):
        super().__init__()
        self.mode = mode
        self.semantic_viz = semantic_viz
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        # Load texture labels from YAML config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if 'labels' in cfg:
            labels = cfg['labels']
        else:
            raise ValueError('Config file must contain "labels".')
        if not isinstance(labels, list):
            raise ValueError('Texture labels in config must be provided as a list.')
        self.labels: List[str] = labels
        self.num_labels: int = len(labels)

        # Map each texture label to an integer ID
        self.label_to_int: Dict[str, int] = {
            label: idx for idx, label in enumerate(self.labels)
        }

        # Prepare label feature vectors
        if self.mode == 'one_hot':
            # One-hot vectors: identity matrix
            label_vectors = torch.eye(self.num_labels)
            self.register_buffer('label_vectors', label_vectors)
        elif self.mode == 'embed':
            if text_encoder is None or text_tokenizer is None:
                raise ValueError('text_encoder must be provided in "embed" mode.')
            embeddings = self.tokenize_labels(self.labels)
            if not isinstance(embeddings, torch.Tensor):
                raise ValueError('text_encoder must return a torch.Tensor.')
            if embeddings.shape[0] != self.num_labels:
                raise ValueError(
                    f'text_encoder returned embeddings for {embeddings.shape[0]} labels, '
                    f'expected {self.num_labels}.'
                )
            self.register_buffer('label_vectors', embeddings)
        else:
            raise ValueError(f'Unsupported mode: {self.mode}. Choose "one_hot" or "embed".')

    def forward(self, features: torch.Tensor, query_label: str = None, raw_intensity: bool = False) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): shape [b, c, h, w]
            query_label (str, optional): if provided, use this label to visualize the feature map.
                Only applicable in 'embed' mode with semantic_viz='query'.
        Returns:
            torch.Tensor: semantic map of shape [b, h, w] with integer labels
        """
        if features.dim() != 4:
            raise ValueError(f'Expected features of shape [b, c, h, w], got {features.shape}.')
        b, c, h, w = features.shape

        if self.semantic_viz == 'feat':
            # Use viz_feat to visualize the feature map directly
            vis_img = viz_feat(features)
            # Convert PIL image to tensor with shape [h, w, 3]
            render = torch.from_numpy(np.array(vis_img)).to(features.device)
            return render
            
        if self.mode == 'one_hot':
            base = torch.eye(self.num_labels, device=features.device)
            if c < self.num_labels:
                raise ValueError(
                    f'Feature dim ({c}) must be ≥ number of labels ({self.num_labels})'
                )
            # Pad zeros on the right to get shape [num_labels, c]
            label_vecs = F.pad(base, (0, c - self.num_labels))
        else:
            if query_label is not None:
                label_vectors = self.tokenize_labels([query_label])
            else:
                label_vectors = self.label_vectors
            # Normalize to ensure they are unit vectors
            label_vectors = label_vectors / label_vectors.norm(dim=1, keepdim=True)
            label_vecs = label_vectors.to(device=features.device, dtype=features.dtype)
    
        features = features / features.norm(dim=1, keepdim=True)
        flattened = features.permute(0,2,3,1).reshape(b, h*w, c)     # [b, h*w, c]
        sims = flattened @ label_vecs.T       # [b, h*w, num_labels]
        
        if self.mode == 'embed' and self.semantic_viz == 'query':
            # Return intensity map for the first label only
            query_sims = sims[:, :, 0]  # Get similarities to first label [b, h*w]
            # Reshape to [b, h, w]
            intensity_map = query_sims.reshape(b, h, w)
            # import ipdb; ipdb.set_trace()
            if raw_intensity:
                # If raw intensity is requested, return the normalized intensity map directly
                render = intensity_map
                return render
            # Normalize to [0, 1] range for better visualization
            intensity_map = (intensity_map - intensity_map.min()) / (intensity_map.max() - intensity_map.min() + 1e-8)
            # Convert to colormap
            intensity_np = intensity_map.detach().cpu().numpy()
            intensity_list = []
            # Apply colormap (jet, viridis, plasma, etc.) by looping over the batch
            for i in range(intensity_np.shape[0]):
                intensity_list.append(plt.cm.jet(intensity_np[i])[:, :, :3])  # Remove alpha channel
            intensity_np = np.stack(intensity_list, axis=0)
            # Convert back to tensor with RGB values
            if intensity_np.shape[0] == 1:
                intensity_np = intensity_np[0]
            render = torch.from_numpy(intensity_np).to(intensity_map.device)
            render = (render * 255).to(torch.uint8)
            
            return render
        else:
            # Default segmentation mode
            idxs = sims.argmax(-1)                # [b, h*w]
            render = idxs.reshape(b, h, w).long()
            render = semantic_to_color(render)
            return render

    def tokenize_labels(self, labels) -> None:
        """
        Tokenizes the texture labels using the provided tokenizer.
        """
        if self.mode != 'embed':
            raise ValueError('tokenize_labels is only applicable in "embed" mode.')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        label_tokens = self.text_tokenizer(labels).to(device)
        with torch.no_grad():
            embeddings = self.text_encoder(label_tokens)
        return embeddings.detach()