import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class StyleGAN2NADATrainer:
    def __init__(
        self,
        generator,
        generator_clean,
        clip_loss,
        optimizer,
        scheduler,
        device,
        source_text,
        target_text,
        num_steps=1000,
        l2_lambda=0.4,
        latent_dim=512,
        visualize_every=200
    ):
        """
        Trainer для StyleGAN-NADA обучения

        Args:
            generator: обучаемый StyleGAN генератор
            generator_clean: исходный frozen генератор
            clip_loss: CLIP loss
            optimizer: optimizer
            scheduler: scheduler
            device: cuda/cpu
            source_text: исходный текстовый домен
            target_text: целевой текстовый домен
        """

        self.generator = generator
        self.generator_clean = generator_clean

        self.clip_loss = clip_loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        self.source_text = source_text
        self.target_text = target_text

        self.num_steps = num_steps

        self.l2_lambda = l2_lambda

        self.latent_dim = latent_dim

        self.visualize_every = visualize_every


        self.losses = {
            "clip": [],
            "l2": [],
            "all": [],
            "step": []
        }


        # frozen generator
        self.generator_clean.eval()

        for param in self.generator_clean.parameters():
            param.requires_grad = False


    def generate_images(self):
        """
        Генерация изображения из случайного latent
        """

        z = torch.randn(1,self.latent_dim,device=self.device)
        w = self.generator.get_latent(z)


        image, _ = self.generator([w], input_is_latent=True, randomize_noise=False)

        with torch.no_grad():
            image_clean, _ = self.generator_clean([w],input_is_latent=True,randomize_noise=False)

        return image, image_clean



    def train_step(self, step):
        """
        Один шаг обучения
        """

        self.optimizer.zero_grad()
        image, image_clean = self.generate_images()

        # CLIP loss
        cur_clip_loss = self.clip_loss(generated_image=image, frozen_image=image_clean, source_text=self.source_text, target_text=self.target_text)

        # L2 similarity loss
        cur_l2_loss = F.mse_loss(image, image_clean)

        # общий loss
        loss = (cur_clip_loss + self.l2_lambda * cur_l2_loss)

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # сохраняем историю
        self.losses["clip"].append(cur_clip_loss.item())
        self.losses["l2"].append(cur_l2_loss.item())
        self.losses["all"].append(loss.item())
        self.losses["step"].append(step)


        return {
            "loss": loss.item(),
            "clip": cur_clip_loss.item(),
            "l2": cur_l2_loss.item()
        }



    def visualize(
        self,
        image,
        image_clean
    ):
        """
        Визуализация результата
        """

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        img = image.detach().cpu()[0].clip(-1,1).permute(1,2,0)
        img = (img + 1) / 2
        img_clean = image_clean.detach().cpu()[0].clip(-1,1).permute(1,2,0)
        img_clean = (img_clean + 1) / 2

        axs[0].imshow(img)
        axs[0].set_title("Generated image")
        axs[0].axis("off")

        axs[1].imshow(img_clean)
        axs[1].set_title("Original image")
        axs[1].axis("off")

        plt.show()



    def train(self):
        """
        Основной цикл обучения
        """
        torch.cuda.empty_cache()
        self.generator.train()

        for step in range(self.num_steps + 1):
            metrics = self.train_step(step)

            if step % self.visualize_every == 0:
                print(f"Step {step} Total loss: {metrics['loss']:.4f} CLIP: {metrics['clip']:.4f} L2: {metrics['l2']:.4f}")

                image, image_clean = self.generate_images()
                self.visualize(image, image_clean)

            torch.cuda.empty_cache()

        return self.losses



    def plot_losses(self):
        """
        Графики обучения
        """
        plt.figure(figsize=(12,5))
        plt.plot(self.losses["step"], self.losses["all"], label="Total")
        plt.plot(self.losses["step"], self.losses["clip"], label="CLIP")
        plt.plot(self.losses["step"], self.losses["l2"], label="L2")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()


    def save_checkpoint(
        self,
        path
    ):
        """
        Сохранение модели
        """
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "losses": self.losses
            },
            path
        )