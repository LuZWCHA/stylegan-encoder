import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import glob
import matplotlib.pyplot as plt

Model = './models/network-snapshot-001640.pkl'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
_Gs_cache = dict()

def main():
    change_style_figure('mixed-figure', 'mix1', 'mix2', Gs,
                        style_ranges=[range(0, 2), range(2, 4), range(4, 6), range(6, 8), range(8,10)])


def load_Gs(model):
    if model not in _Gs_cache:
        model_file = glob.glob(Model)
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')

        _G, _D, Gs = pickle.load(model_file)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

        # Print network details.
        Gs.print_layers()

        _Gs_cache[model] = Gs
    return _Gs_cache[model]


def change_style_figure(save_name, mix1, mix2, Gs, style_ranges):
    os.makedirs(config.generated_dir, exist_ok=True)
    save_path = os.path.join(config.generated_dir, save_name + '.png')
    print(save_path)

    os.makedirs(config.dlatents_dir, exist_ok=True)
    src = np.load(os.path.join(config.dlatents_dir, mix1 + '.npy'))
    dst = np.load(os.path.join(config.dlatents_dir, mix2 + '.npy'))

    src_dlatents = np.expand_dims(src, axis=0)
    dst_dlatents = np.expand_dims(dst, axis=0)

    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=config.randomize_noise, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=config.randomize_noise, **synthesis_kwargs)

    Style_No = len(style_ranges)
    w = src_images.shape[1]
    h = src_images.shape[2]
    canvas = PIL.Image.new('RGBA', (w * (Style_No + 2), h))

    canvas.paste(PIL.Image.fromarray(src_images[0], 'RGBA'), (0, 0))

    canvas.paste(PIL.Image.fromarray(dst_images[0], 'RGBA'), ((Style_No + 1) * w, 0))

    row_dlatents = np.stack([src_dlatents[0]] * Style_No)

    for i in range(Style_No):
        row_dlatents[i, style_ranges[i]] = dst_dlatents[0, style_ranges[i]]

    row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=config.randomize_noise, **synthesis_kwargs)

    for col, image in enumerate(list(row_images)):
        canvas.paste(PIL.Image.fromarray(image, 'RGBA'), ((col + 1) * w, 0))

    canvas.show()
    canvas.save(save_path)


def move_and_show(latent_vector, direction, coeffs, Gs):
    fig, ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
        img_array = Gs.components.synthesis.run(np.expand_dims(new_latent_vector, axis=0), randomize_noise=config.randomize_noise, **synthesis_kwargs)
        img = PIL.Image.fromarray(img_array[0], 'RGBA')
        ax[i].imshow(img)
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    plt.show()

# init
tflib.init_tf()
Gs = load_Gs(Model)

if __name__ == "__main__":
    main()