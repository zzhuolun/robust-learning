import os
import torchvision
from torchvision import transforms


def convert_to_range(value):
  if isinstance(value, list):
    return value
  return [value]

def write_img(img, path, img_info = {}):
  print("Saving image in path", path)
  #img = torchvision.utils.make_grid(img.data, normalize=True)
  img = transforms.ToPILImage()(img.cpu()).convert("RGB")
  if isinstance(img_info, dict) and 'height' in img_info.keys() and 'width' in img_info.keys():
    img = img.resize((img_info['width'], img_info['height']), resample = 0)
  img.save(path)

def save_adversarial_images(result, output_dir, typ, names, image_info = None ):
  for x, data in result.items():
    # The structure for PGD images has two levels as opposed to one in others
    if typ=="pgd":
      for eps, images in data.items():
        # Create the directory to store images
        x   = str(x).replace('.', '')
        eps = str(eps).replace('.', '')
        output_path = os.path.join(output_dir, typ, x, eps)
        if not os.path.exists(output_path):
          os.makedirs(output_path, exist_ok=True)

        for index, adv_img in enumerate(images):
          img_path = os.path.join( output_path, os.path.basename(names[index]) )
          write_img(adv_img, img_path, image_info)
          
    # The structure for noisy and FGSM images is the same
    else:
      # Create the directory to store images
      x = str(x).replace('.', '')
      output_path = os.path.join(output_dir, typ, x)
      if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

      for index, adv_img in enumerate(data):
        img_path = os.path.join( output_path, os.path.basename(names[index]) )
        write_img(adv_img, img_path, image_info)

