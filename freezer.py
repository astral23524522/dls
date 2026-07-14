import torch
import torch.nn as nn


class LayerFreezer:

    def __init__(self, generator, generator_frozen, clip_loss, device="cuda"):
        self.generator = generator
        self.generator_frozen = generator_frozen
        self.clip_loss = clip_loss
        self.device = device

    # UTILITIES
    def _freeze_all(self):
        """
        Freeze every parameter.
        """
        for p in self.generator.parameters():
            p.requires_grad = False

    def _unfreeze_all(self):
        """
        Unfreeze every parameter.
        """
        for p in self.generator.parameters():
            p.requires_grad = True

    def _trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.generator.parameters())

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.generator.parameters())
        trainable = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)

        print("=" * 60)
        print(f"Trainable: {trainable:,}")
        print(f"Total:      {total:,}")
        print(f"Percent:    {100 * trainable / total:.2f}%")
        print("=" * 60)

        for name, p in self.generator.named_parameters():
            if p.requires_grad:
                print(name)

    # LATENT HELPERS

    def _sample_w_plus(self, batch_size=2):
        """
        Generate random W+ latent.
        """
        latent_dim = self.generator_frozen.style_dim
        latent_z = torch.randn(batch_size, latent_dim, device=self.device)

        with torch.no_grad():
            latent_w = self.generator_frozen.style(latent_z)

        latent_w_plus = latent_w.unsqueeze(1).repeat(1, self.generator_frozen.n_latent, 1)

        return latent_w_plus

    def _build_trainable_w_plus(self, batch_size=2):
        latent = self._sample_w_plus(batch_size)
        latent = latent.detach().clone()
        latent.requires_grad_(True)

        return latent

    # MODULE HELPERS

    def _candidate_modules(self):
        """
        Collect all StyledConv + ToRGB modules.
        """
        modules = []

        for module in self.generator.modules():
            name = module.__class__.__name__
            if name in ["StyledConv", "ToRGB"]:
                modules.append(module)

        return modules

    def _module_names(self, modules):
        names = []

        for name, module in self.generator.named_modules():
            if module in modules:
                names.append(name)

        return names

    def _unfreeze_modules(self, module_names):
        self._freeze_all()
        for name, p in self.generator.named_parameters():
            if any(name.startswith(module_name) for module_name in module_names):
                p.requires_grad = True

    # ACTIVE LAYER SELECTION (W+ OPTIMIZATION)

    def _find_active_modules(self, text_features, k=5, batch_size=2, lr=1e-2, iterations=3, ):
        """
        Returns top-k most active StyledConv / ToRGB modules.
        """
        latent = self._build_trainable_w_plus(batch_size)
        latent_initial = latent.detach().clone()
        optimizer = torch.optim.Adam([latent], lr=lr)

        self.generator.eval()

        for _ in range(iterations):
            optimizer.zero_grad()
            generated, _ = self.generator([latent], input_is_latent=True, randomize_noise=False)
            loss = self.clip_loss(generated, text_features)
            loss.backward()
            optimizer.step()

        involvement = (torch.abs(latent - latent_initial).mean(dim=-1).mean(dim=0))
        indices = torch.topk(involvement, k).indices.cpu().tolist()

        modules = self._candidate_modules()
        selected = [modules[i] for i in indices]

        return selected

    # MANUAL FREEZE

    def freeze_manual(self, freeze_mapping=True, freeze_initial_blocks=2, freeze_torgb=False):
        """
        Manual StyleGAN freezing.

        Parameters
        ----------
        freeze_mapping : bool
            Freeze mapping network.

        freeze_initial_blocks : int
            Number of initial resolution blocks to freeze.

        freeze_torgb : bool
            Freeze every ToRGB layer.
        """

        self._unfreeze_all()

        for name, param in self.generator.named_parameters():
            # Mapping
            if freeze_mapping:
                if name.startswith("style."):
                    param.requires_grad = False

            # First block (4x4)

            if freeze_initial_blocks >= 1:
                if name.startswith("conv1.") or name.startswith("to_rgb1."):
                    param.requires_grad = False

            # Remaining StyledConv blocks
            for i in range(freeze_initial_blocks - 1):

                if (name.startswith(f"convs.{2 * i}.") or name.startswith(f"convs.{2 * i + 1}.")
                        or name.startswith(f"to_rgbs.{i}.")):
                    param.requires_grad = False

            # ToRGB
            if freeze_torgb:
                if name.startswith("to_rgb1.") or name.startswith("to_rgbs."):
                    param.requires_grad = False

    # ADAPTIVE FREEZE
    def freeze_adaptive(self, text_features, k=5, batch_size=2, lr=1e-2, iterations=3, verbose=True,):
        """
        Adaptive layer selection.

        Finds k most active StyledConv / ToRGB blocks
        using W+ optimization.
        """

        selected_modules = self._find_active_modules(text_features=text_features, k=k, batch_size=batch_size, lr=lr,
                                                     iterations=iterations)
        module_names = self._module_names(selected_modules)
        self._unfreeze_modules(module_names)

        if verbose:
            print()
            print("=" * 60)
            print("Selected modules")

            for name in module_names:
                print(name)

            print("=" * 60)
