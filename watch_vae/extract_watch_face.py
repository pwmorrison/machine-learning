
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from glob import glob
from pathlib import Path

"""
Using some basic edge analysis to extract square images of the faces of the watches.
The purpose is to create a dataset to train a Spatial Transformer Network (STN) that finds the faces in an image, and
transforms the image to crop out the face. The result would be better for the watch VAE.
Requires a manual pass to delete watches that still aren't great.

TODO:
23, 133 - White watch on white background. Enhance contrast to detect edges.
"""



def extract_face_square(img):
    """
    Finds a square in the given image that contains the face of the watch.
    Assumptions:
    * Grayscale image.
    * The background is uniform (no edges, since we find the first and last edge on each row).
    * The face has a significant contrast with the background (since we find edges).
    * The face is front-on.
    * The face is circular or rectangular (we extract a square).
    """
    # Extract edges.
    edges = cv2.Canny(img, 50, 200)

    # Remove edges at the top and bottom, for image 103, 103, 134, 203, 242, 262, ...
    edges[:10, :] = 0
    edges[-10:, :] = 0

    # Find the middle row of pixels.
    # Find the row with the largest distance between left and right edge.
    # Find the maximum distance between left and right edge.
    row_dists = []
    row_nums = np.array(range(edges.shape[0]))
    for row_num in row_nums:
        row = edges[row_num, :]
        edge_indices = np.array(range(len(row)))[row != 0]
        if len(edge_indices) < 2:
            dist = 0
        else:
            dist = edge_indices[-1] - edge_indices[0]
        row_dists.append(dist)
    #max_dist = np.amax(row_dists)
    # Get the rows that have a large distance, not necessarily the max.
    max_dist = np.percentile(row_dists, 95)
    max_dist_rows = row_nums[row_dists >= max_dist]
    # Choose the middle row that has a maximum distance.
    #middle_row_num = int(np.median(max_dist_rows))
    middle_row_num = np.amin(max_dist_rows) + (np.amax(max_dist_rows) - np.amin(max_dist_rows)) // 2
    # middle_row_num = edges.shape[0] // 2

    # Get the row that has the max dist.
    # Use this row to extract the left and right edge.
    max_dist_row_nums = row_nums[row_dists == np.amax(row_dists)]
    max_dist_row_num = int(np.median(max_dist_row_nums))
    max_dist_row = edges[max_dist_row_num, :]

    # Get the middle row of pixels.
    middle_row = edges[middle_row_num, :]
    # Get the indices of edges in the row.
    edge_indices = np.array(range(len(max_dist_row)))[max_dist_row != 0]
    # Get the bounding box, using the first and last edges in the middle row.
    left = edge_indices[0]
    right = edge_indices[-1]
    width = right - left
    top = middle_row_num - width // 2
    bottom = top + width

    if 0:
        fig, axes = plt.subplots(1, 3, squeeze=False)
        # Input image.
        ax = axes[0, 0]
        ax.imshow(img)
        # Edges and rect.
        ax = axes[0, 1]
        ax.imshow(edges)
        ax.scatter([left, right], [middle_row_num, middle_row_num])
        rect = patches.Rectangle((left, top), width, width, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Middle row.
        ax = axes[0, 2]
        ax.plot(middle_row)
        plt.show()

    return top, bottom, left, right


def main():
    filename = r'D:\data\a_dataset_of_watches\watches\watches\images\1019.jpg'
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top, bottom, left, right = extract_face_square(img_gray)
    width = right - left

    if 1:
        fig, axes = plt.subplots(1, 1, squeeze=False)
        # Input image.
        ax = axes[0, 0]
        ax.imshow(img[..., ::-1])
        rect = patches.Rectangle((left, top), width, width, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()


def main_batch():
    image_path = Path(r'D:\data\a_dataset_of_watches\watches\watches\images')
    out_path = Path(r'D:\data\a_dataset_of_watches\watches\watches\images_faces')
    images_filenames = glob(str(image_path / '*.jpg'))
    #print(images_filenames)

    out_path.mkdir(exist_ok=True)

    for filename in images_filenames:
        print(filename)
        try:
            img = cv2.imread(filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            top, bottom, left, right = extract_face_square(img_gray)
            width = right - left

            face_img = img[top: bottom, left: right, :]

            id = Path(filename).stem

            if 1:
                # Output the face image.
                cv2.imwrite(str(out_path / f'{id}.jpg'), face_img)

            if 0:
                fig, axes = plt.subplots(1, 2, squeeze=False)
                # Input image.
                ax = axes[0, 0]
                ax.imshow(img[..., ::-1])
                rect = patches.Rectangle((left, top), width, width, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.set_title(id)

                ax = axes[0, 1]
                ax.imshow(face_img[..., ::-1])

                plt.show()
        except:
            print(f'FAILED {filename}')



if __name__ == '__main__':
    #main()
    main_batch()
