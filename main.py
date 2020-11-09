from dcgan import *



# Learning Parameters
CRITERION = nn.BCEWithLogitsLoss()
N_EPOCHS = 50
Z_DIM = 64
DISPLAY_STEP = 500
BATCH_SIZE = 128
LR = 0.0002
DEVICE = 'cuda'
DOWNLOAD_DATA = True

# Optimization-Momentur Parameters
BETA_1 = 0.5
BETA_2 = 0.999


# Normalize images - values between -1 & 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])
# Data loader object
dataloader = DataLoader(
    MNIST('.', download=DOWNLOAD_DATA, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True)


# Instantiate Generator & Discriminator
gen = Generator(Z_DIM).to(DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LR, betas=(BETA_1, BETA_2))
disc = Discriminator().to(DEVICE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LR, betas=(BETA_1, BETA_2))

# Initialize wieghts to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
    

def run_neural_network():
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    for epoch in range(N_EPOCHS):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(DEVICE)

            ## Update Discriminator
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, Z_DIM, DEVICE)
            fake = gen(fake_noise).detach()
            disc_fake_pred = disc(fake)
            disc_fake_loss = CRITERION(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            real_disc_pred = disc(real)
            real_disc_loss = CRITERION(real_disc_pred, torch.ones_like(real_disc_pred))
            disc_loss = (disc_fake_loss + real_disc_loss) / 2
            # Update Gradients
            disc_loss.backward(retain_graph=True)
            # Update Optimizer
            disc_opt.step()
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss / DISPLAY_STEP

            ## Update Generator
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, Z_DIM, DEVICE)
            fake_2 = gen(fake_noise_2)
            disc_fake_2_pred = disc(fake_2)
            gen_loss = CRITERION(disc_fake_2_pred, torch.ones_like(disc_fake_2_pred))
            # Update gradients
            gen_loss.backward()
            # Update optimizer
            gen_opt.step()
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / DISPLAY_STEP

            ## Visualizing
            if (cur_step % DISPLAY_STEP) == 0 and (cur_step > 0):
                print(f"Step: {cur_step}  Generator loss: {mean_generator_loss}  Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0 

            cur_step += 1