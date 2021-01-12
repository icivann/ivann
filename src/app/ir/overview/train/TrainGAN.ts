import { nodeName } from '@/app/ir/irCommon';
import { TrainGANOptions } from '@/nodes/overview/train/TrainGAN';

export default class TrainGAN {
  constructor(
    public readonly name: string,
    public readonly Epochs: number,
    public readonly Device: string,
    public readonly LogInterval: number,
    public readonly RealLabel: number,
    public readonly FakeLabel: number,
  ) {
  }

  static build(options: Map<string, any>): TrainGAN {
    return new TrainGAN(
      options.get(nodeName),
      options.get(TrainGANOptions.Epochs),
      options.get(TrainGANOptions.Device),
      options.get(TrainGANOptions.LogInterval),
      options.get(TrainGANOptions.RealLabel),
      options.get(TrainGANOptions.FakeLabel),
    );
  }

  public initCode(): string[] {
    return [`
def train_gan(model_g, model_d, train_loader, test_loader, optimizer_g, optimizer_d, loss_f, device, epochs, log_interval=${this.LogInterval}, fake_label, real_label):
  img_list = []
  G_losses = []
  D_losses = []
  iters = 0

  # Lists to keep track of progress
  img_list = []
  G_losses = []
  D_losses = []
  iters = 0

  print("Starting Training Loop...")
  # For each epoch
  for epoch in range(epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      ## Train with all-real batch
      model_D.zero_grad()
      # Format batch
      real_cpu = data[0].to(device)
      b_size = real_cpu.size(0)
      label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
      # Forward pass real batch through D
      output = model_D(real_cpu).view(-1)
      # Calculate loss on all-real batch
      errD_real = loss_f(output, label)
      # Calculate gradients for D in backward pass
      errD_real.backward()
      D_x = output.mean().item()

      ## Train with all-fake batch
      # Generate batch of latent vectors
      noise = torch.randn(b_size, nz, 1, 1, device=device)
      # Generate fake image batch with G
      fake = model_G(noise)
      label.fill_(fake_label)
      # Classify all fake batch with D
      output = model_D(fake.detach()).view(-1)
      # Calculate D's loss on the all-fake batch
      errD_fake = loss_f(output, label)
      # Calculate the gradients for this batch
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      # Add the gradients from the all-real and all-fake batches
      errD = errD_real + errD_fake
      # Update D
      optimizerD.step()

      ############################
      # (2) Update G network: maximize log(D(G(z)))
      ###########################
      model_G.zero_grad()
      label.fill_(real_label)  # fake labels are real for generator cost
      # Since we just updated D, perform another forward pass of all-fake batch through D
      output = model_D(fake).view(-1)
      # Calculate G's loss based on this output
      errG = loss_f(output, label)
      # Calculate gradients for G
      errG.backward()
      D_G_z2 = output.mean().item()
      # Update G
      optimizerG.step()

      # Output training stats
      if i % 50 == 0:
        print('[%d/%d][%d/%d]\\tLoss_D: %.4f\\tLoss_G: %.4f\\tD(x): %.4f\\tD(G(z)): %.4f / %.4f'
          % (epoch, num_epochs, i, len(dataloader),
             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

      # Save Losses for plotting later
      G_losses.append(errG.item())
      D_losses.append(errD.item())

      # Check how the generator is doing by saving G's output on fixed_noise
      if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        with torch.no_grad():
          fake = model_G(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

      iters += 1
`.trim()];
  }

  public callCode(params: string[]): string[] {
    return [`train_gan(${params.join(', ')}, device="${this.Device}", epoch=${this.Epochs}, log_interval=${this.LogInterval}, fake_label=${this.FakeLabel}, real_label=${this.RealLabel})`];
  }
}
