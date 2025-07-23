# -*- coding: utf-8 -*-
"""
Culvert Instance Segmentation Pipeline
- Performs semantic segmentation on rock images
- Applies color-based clustering for instance segmentation
- Combines results for final output
"""

# ===== Environment Setup =====
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # Fix CUDA initialization
os.environ['KERAS_BACKEND'] = 'tensorflow'  # Ensure proper backend
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Prevent Qt conflicts

# ===== Import Dependencies =====
import sys
import numpy as np
import cv2
import torch
import torchvision
import tensorflow as tf
import time
import argparse
import logging

# Add current directory to path for local module imports
sys.path.insert(0, '.')

# Suppress non-critical warnings
logging.getLogger().setLevel(logging.ERROR)
torch.set_grad_enabled(False)

# ===== Custom DepthwiseConv2D Layer =====
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove problematic parameter
        super().__init__(*args, **kwargs)

# ===== Configuration =====
# Update these paths to match your system
img_path = '/home/ksim/Desktop/EXPLORE/OFFSEG/img'
final_path = '/home/ksim/Desktop/EXPLORE/OFFSEG'
seg_model_path = 'model_final.pth'
class_model_path = 'keras_model.h5'

# ===== Segmentation Functions =====
def to_tensor(im, mean=(0.3257, 0.3690, 0.3223), std=(0.2112, 0.2148, 0.2115)):
    """Convert image to normalized tensor"""
    # Convert to tensor and normalize
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    return transform(im)

def img_seg(im, net):
    """Perform semantic segmentation using BiSeNet"""
    # Preprocess and run through model
    im_tensor = to_tensor(im).unsqueeze(0).cuda()
    out = net(im_tensor)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    
    # Create color-coded segmentation map
    pred = pal[out]
    return pred, out

def palette_lst(masked_img, n_classes=4):
    """K-means clustering using OpenCV for color segmentation"""
    height, width = masked_img.shape[:2]
    data = masked_img.reshape(-1, 3).astype(np.float32)
    
    # K-means parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        data, n_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Process results
    centers = np.uint8(centers)
    clustered = centers[labels.flatten()].reshape(height, width, 3)
    
    # Create list of segmented images
    img_lst = []
    for i in range(n_classes):
        mask = np.all(clustered == centers[i], axis=-1)
        segmented = np.zeros_like(masked_img)
        segmented[mask] = masked_img[mask]
        img_lst.append(segmented)
    
    return img_lst

def trav_cut(img, lpool):
    """Extract traversable section based on segmentation mask"""
    # Process segmentation mask
    lpool = (lpool != 1).astype(np.uint8) * 255
    lpool = cv2.resize(lpool, (img.shape[1], img.shape[0]))
    
    # Create transparent overlay
    image_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = (lpool == 0)
    image_bgra[..., 3] = np.where(mask, 0, 255)
    
    return cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)

def mask_pred(img_lst, model):
    """Predict mask classes using classification model"""
    mask_class = []
    for img in img_lst:
        # Preprocess image
        im = cv2.resize(img, (224, 224))
        im = (im / 127.0) - 1  # Normalize to [-1, 1]
        
        # Predict class
        prediction = model.predict(im[np.newaxis, ...], verbose=0)
        mask_class.append(np.argmax(prediction) + 4)  # Offset for class IDs
    
    return mask_class

def mask_comb(newpool, img_lst, mask_class):
    """Combine masks into final segmentation map"""
    # Convert base pool to grayscale
    newpool_gray = cv2.cvtColor(newpool.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    for i, (img, cls_val) in enumerate(zip(img_lst, mask_class)):
        # Resize segmented image
        ima = cv2.resize(img, (newpool_gray.shape[1], newpool_gray.shape[0]))
        
        # Convert to grayscale and apply class value
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        segmented = np.zeros_like(newpool_gray)
        segmented[mask] = cls_val
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)
        
        # Combine masks
        newpool_gray = np.where(segmented > 0, segmented, newpool_gray)
    
    return newpool_gray

def col_seg(image, pool, model):
    """Main color segmentation pipeline"""
    # Extract traversable regions
    travcut = trav_cut(image, pool)
    
    # Apply k-means clustering
    msk_img = palette_lst(travcut)
    
    # Predict classes for each segment
    predicts = mask_pred(msk_img, model)
    
    return msk_img, predicts

# ===== Main Execution =====
if __name__ == "__main__":
    # Parse arguments
    parse = argparse.ArgumentParser(description='Rock Instance Segmentation Pipeline')
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2', help='Segmentation model type')
    parse.add_argument('--seg-weight', type=str, default=seg_model_path, help='Segmentation model weights path')
    parse.add_argument('--class-weight', type=str, default=class_model_path, help='Classification model path')
    parse.add_argument('--img-dir', type=str, default=img_path, help='Input images directory')
    parse.add_argument('--out-dir', type=str, default=final_path, help='Output directory')
    args = parse.parse_args()
    
    # Load classification model
    print("\n===== Loading Classification Model =====")
    try:
        model = tf.keras.models.load_model(
            args.class_weight,
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
        )
        print(f"Classification model loaded successfully from {args.class_weight}")
    except Exception as e:
        print(f"Error loading classification model: {e}")
        sys.exit(1)
    
    # Load segmentation model
    print("\n===== Loading Segmentation Model =====")
    try:
        # Import model architecture components
        import lib.transform_cv2 as T
        from lib.models import model_factory
        from configs import cfg_factory
        
        # Create model architecture
        cfg = cfg_factory[args.model]
        net = model_factory[cfg.model_type](4)  # 4 classes
        
        # Load weights
        state_dict = torch.load(args.seg_weight, map_location='cpu')
        net.load_state_dict(state_dict)
        
        # Move to GPU and set evaluation mode
        net = net.cuda().eval()
        print(f"Segmentation model loaded successfully from {args.seg_weight}")
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        sys.exit(1)
    
    # Create color palette
    pal = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
    
    # Process images
    img_list = sorted(os.listdir(args.img_dir))
    print(f"\n===== Processing {len(img_list)} Images =====")
    
    for i, img_name in enumerate(img_list):
        print(f"\nProcessing image {i+1}/{len(img_list)}: {img_name}")
        start_time = time.time()
        
        try:
            # Load image
            image_path = os.path.join(args.img_dir, img_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            
            # Convert to RGB for segmentation model
            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform semantic segmentation
            pred, pool = img_seg(im_rgb, net)
            cv2.imwrite(os.path.join(args.out_dir, f"seg_{img_name}"), pred)
            
            # Perform color-based instance segmentation
            msk_img, predicts = col_seg(image, pool, model)
            
            # Combine masks into final output
            final_pool = mask_comb(pred, msk_img, predicts)
            
            # Save results
            output_img = pal[final_pool]
            cv2.imwrite(os.path.join(args.out_dir, f"final_{img_name}"), output_img)
            
            print(f"Completed in {time.time()-start_time:.2f} seconds")
            print(f"Results saved to {args.out_dir}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
    
    print("\n===== Processing Complete =====")
    print(f"All images processed successfully. Outputs in {args.out_dir}")
