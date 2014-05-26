Introduction
===============

five steps:

Find Harris corners in two images

Extract SIFT descriptors for those keypoints

Match keypoints

Calculate homography using RANSAC

Apply the homography to the second image, so that if the two images were on top of one another, their features would be aligned.
