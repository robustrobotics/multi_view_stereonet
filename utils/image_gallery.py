# Copyright 2021 Massachusetts Institute of Technology
#
# @file image_gallery.py
# @author W. Nicholas Greene
# @date 2020-07-02 23:44:46 (Thu)

import os
import argparse

def create_simple_gallery(image_dir, num_per_row=3, output_file="index.html", title="Image Gallery"):
    """Create a simple gallery with num_per_row images per row.
    """
    # Grab all images.
    images = []
    for root, dirs, files in os.walk(image_dir):
        for filename in sorted(files):
            filename_full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(filename_full_path, image_dir)
            if filename_full_path.endswith(".png") or filename_full_path.endswith(".jpg"):
                images.append(rel_path)
    images = sorted(images)

    # Write html file.
    html_file = os.path.join(image_dir, output_file)
    with open(html_file, "w") as target:
        target.write("<html><head><title>{}</title></head><body><center>\n".format(title))

        for image in images:
            image_str = "<a href={}><img src=\"{}\" style=\"float: left; width: {}%; image-rendering: pixelated\"></a>\n".format(image, image, 100.0 / num_per_row)
            target.write(image_str)

        target.write("</center></body></html>\n")

    return

def create_training_gallery(image_dir, image_height_pix=256, output_file="index.html", title="Image Gallery", delim="_"):
    """Create a gallery where each rows shows the evolution of an image during training.

    Assumes images are in the following format:
      <image_id>_<epoch>_<step>.jpg

    Epoch and step are optional, but if provided must be zero padded so sorting
    will put them in the appropriate order.
    """
    # Grab all images.
    id_to_images = {}
    for root, dirs, files in os.walk(image_dir):
        for filename in sorted(files):
            filename_full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(filename_full_path, image_dir)
            if filename_full_path.endswith(".png") or filename_full_path.endswith(".jpg"):
                tokens = os.path.splitext(os.path.basename(rel_path))[0].split(delim)
                image_id = tokens[0]

                if image_id not in id_to_images:
                    id_to_images[image_id] = []

                id_to_images[image_id].append(rel_path)

    for image_id, images in id_to_images.items():
        id_to_images[image_id] = sorted(images, reverse=True)

    # Write html file.
    html_file = os.path.join(image_dir, output_file)
    with open(html_file, "w") as target:
        target.write("<html><head><title>{}</title></head><body>\n".format(title))
        target.write("<table>\n")

        for image_id, images in id_to_images.items():
            target.write("<tr align=\"left\">\n")
            for image in images:
                image_str = "<td><a href={}><img src=\"{}\" style=\"height: {}; image-rendering: pixelated\"></a></td>\n".format(
                    image, image, image_height_pix)
                target.write(image_str)
            target.write("</tr>\n")

        target.write("</table>\n")
        target.write("</body></html>\n")

    return

def main():
    # Parse args.
    parser = argparse.ArgumentParser(description="Create simple image gallery from a folder of images.")
    parser.add_argument("image_dir", help="Path to image directory.")
    parser.add_argument("--num_per_row", type=int, default=3, help="Number of images per row.")
    parser.add_argument("--output_file", default="index.html", help="Output file name.")
    parser.add_argument("--title", default="Image Gallery", help="Gallery name.")
    args = parser.parse_args()

    create_simple_gallery(args.image_dir, args.num_per_row, args.output_file, args.title)

    return

if __name__ == '__main__':
    main()
