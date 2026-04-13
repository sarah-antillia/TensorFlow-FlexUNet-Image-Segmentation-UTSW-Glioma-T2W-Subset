<h2>TensorFlow-FlexUNet-Image-Segmentation-UTSW-Glioma-T2W-Subset (2026/04/13)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>UTSW (The University of Texas
Southwestern Medical Center) Glioma T2W</b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>), and a 512x512 pixels upscaled PNG
 <a href="https://drive.google.com/file/d/1IwBEq6PM9pV2OROd2JyBwA5bSMIrXgtS/view?usp=sharing">
UTSW-Glioma-T2W-ImageMask-Subset.zip</a> with colorized masks 
(<a href="https://www.mit.edu/~amini/LICENSE.md">MIT</a>), which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/suvadipchakraborty/utsw-glioma-2d-mri-dataset">
UTSW_Glioma_2D MRI dataset
</a> on the kaggle.com
<br><br>
<hr>
<b>Actual Image Segmentation for UTSW-Glioma-T2W Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the 
ground truth masks, but they lack precision in certain areas.
<br><br>
<b>class_color_map = {NCR (Necrotic Tumor Core):red, ED (Edema):green, ET (Enhancing Tumor):blue}</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10034_11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10034_11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10034_11.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10100_34.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10100_34.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10100_34.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10103_18.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10103_18.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10103_18.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was taken from <br><br>
<a href="https://www.kaggle.com/datasets/suvadipchakraborty/utsw-glioma-2d-mri-dataset">
UTSW_Glioma_2D MRI dataset
</a> on the kaggle.com
<br><br>
For more information of <b>UTSW Glioma 3D MRI</b> datasset, please refer to 
<a href="https://www.cancerimagingarchive.net/collection/utsw-glioma/">
<b>
UTSW-Glioma | The University of Texas Southwestern Glioma MRI dataset with molecular marker characterization and segmentations
</b></a> on <b>The Cancer Imaging Archive (TCIA)</b>.
<br><br> 
The following explanation was taken from the above web site.<br><br>
<b>About Dataset</b><br>
<b>UTSW-Glioma 2D Selected Subset: Technical Specifications</b><br>
<b>1. Overview</b><br>
This dataset is a highly curated subset of the UTSW-Glioma axial 2D MRI collection. 
It was designed to provide a perfectly balanced and spatially aligned 
multi-modal dataset for deep learning tasks such as segmentation, super-resolution, and tumor grade classification.
<br><br>
<b>2. Dataset Composition</b><br>
<ul>
<li><b>Total Patient Cohort:</b> 625 patients.</li>
<li><b>TImage Resolution:</b> 240 x 240 pixels (Uint8 PNG).</li>
<li><b>Z-Axis Subset:</b> Slices 053 to 100 (selected for optimal tumor representation).</li>
<li><b>Slices per Patient:</b> 48 slices across all 5 modalities/masks.</li>
<li><b>Total Validated Files:</b> 150,000 images.</li>
</ul>
<br>
<b>3. Folder Structure</b><br>
<pre>
dataset/BTXXXX/
├── brain_flair/            # Fluid-Attenuated Inversion Recovery
├── brain_t1/               # T1-weighted original
├── brain_t1ce/             # T1-weighted Contrast-Enhanced
├── brain_t2/               # T2-weighted original
└── segmentation_mask/      # Unified segmentation truth
</pre>
<br>
<b>4. Segmentation Mask (The "Truth")</b><br>
To ensure 100% coverage, the segmentation_mask folder uses a priority-based selection from the source data:<br>
1.<b>Manual Resliced (rtumorseg)</b>:Expert human corrections aligned to the 240x240 grid. (Most common)<br>
2.<b>Manual Original</b>: Expert human corrections (Raw grid).<br>
3.<b>FeTS Automated</b>: AI-generated baseline (used only when manual masks are unavailable).<br>
<br>
<b>Label Map (Scaled for Visibility)</b><br>
<table border="1" style="border-collapse: collapse;">
<tr><th>Value</th><th>Label Name</th><th>Anatomy</th>
</tr>
<tr>
<td>0</td><td>Backgroud</td><td>Non-tumor tissue / Air</td>
</tr>
<tr>
<td>50</td><td>Label 1</td><td>NCR(Necrotic tumor core)</td>
</tr>
<tr>
<td>100</td><td>Label 2</td><td>ED (Peritumoral edema)</td>
</tr>
<tr>
<td>200</td><td>Label 4</td><td>ET (Enhancing Tumor)</td>
</tr>
</table>
<br>
<b>License</b><br>
<a href="https://www.mit.edu/~amini/LICENSE.md">
MIT</a>
<br>
<br>
<h3>2 UTSW-Glioma-T2W ImageMask Dataset</h3>
<h3>2.1 Download UTSW-Glioma-T2W Dataset</h3>
 If you would like to train this UTSW-Glioma-T2W Segmentation model by yourself,
 please download the dataset from the google drive our downscaled 
 <a href="https://drive.google.com/file/d/1IwBEq6PM9pV2OROd2JyBwA5bSMIrXgtS/view?usp=sharing">
UTSW-Glioma-T2W-ImageMask-Subset.zip</a> .zip</a> 
(<a href="https://www.mit.edu/~amini/LICENSE.md">MIT</a>) 
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be:
<br>
<pre>
./dataset
└─UTSW-Glioma-T2W
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>UTSW-Glioma-T2W Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/UTSW-Glioma-T2W_Statistics.png" width="512" height="auto"><br>
<br><br>
As shown above, the number of images of train and valid datasets is large enough to use for the
 training set of our segmentation model.
<br><br>
<h4>2.2 Derivation of 512x512 pixels ImageMask Subset</h4>
The folder structure of <b>UTSW_Glioma_2D</b> is the following.<br>
<pre>
./UTSW_Glioma_2D
└─dataset
     ├─BT0001
     │   ├─brain_flair
     │   ├─brain_t1
     │   ├─brain_t1ce
     │   ├─brain_t2
     │   └─segmentation_mask
...
     └─BT1312
          ├─brain_flair
          ├─brain_t1
          ├─brain_t1ce
          ├─brain_t2
          └─segmentation_mask
</pre>
We used a simple Python script and the following class-color-mapping table to generate our PNG T2W dataset with colorized masks 
from PNG files in <b>brain_t2</b> and <b>segmentation_mask</b> folders in half of the original dataset.
<br>
<br>
<table border="1" style="border-collapse: collapse;">
<tr><th>Index</th><th>Category</th><th>Color </th><th>RGB triplet</th></tr>
<tr>
<td>1</td><td>NCR(Necrotic Tumor Core)</td><td>red</td><td>(255,0,0)</td>
</tr>
<tr>
<td>2</td><td>ED (Peritumoral Edema)</td><td>green</td><td>(0,255,0)</td>
</tr>
<tr>
<td>3</td><td>ET (Enhancing Tumor)</td><td>blue</td><td>(0,0,255)</td>
</tr>
</table>
<br>
For simplicity, we excluded all empty black masks and their corresponding images to generate our PNG dataset,
which were irrelevant to train our segmentation model, and upscaled all images and masks to 512x512 pixels from 
the original 240x240 pixels.
<br>
<h3>2.3 Train Image Mask Samples</h3>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>3 Train TensorFlowFlexUNet Model</h3>
 We trained UTSW-Glioma-T2W TensorFlowFlexUNet Model by using the 
<a href="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 4
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 5
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specifed rgb color map dict for UTSW-Glioma-T2W 1+3 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;UTSW-Glioma-T2W rgb color map dict for 1+3 classes.
rgb_map = {(0,0,0):0, (255,0,0):1, (0,255,0):2, (0,0,255):3,}
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results. However, as training progressed through the epochs, the predictions gradually improved. 
<br><br> 
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 48,49,50)</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 50.<br><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/train_console_output_at_epoch50.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/eval/train_losses.png" width="520" height="auto">
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W</b> folder, 
and run the following bat file to evaluate TensorFlowUNet model for UTSW-Glioma-T2W.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/evaluate_console_output_at_epoch50.png" width="1024" height="auto">
<br><br>Image-Segmentation-UTSW-Glioma-T2W
<a href="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this <b>UTSW-Glioma-T2W/test</b> was low, and dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0125
dice_coef_multiclass,0.9937
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W</b> folder, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for UTSW-Glioma-T2W.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of UTSW-Glioma-T2W Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ,
ground truth masks, but they lack precision in certain areas.
<br><br>
<b>class_color_map = {NCR (Necrotic Tumor Core):red, ED (Edema):green, ET (Enhancing Tumor):blue}</b>
<br><br>
<table>
<tr>
<th>Input:Image</th>
<th>Mask (ground_truth)</th>
<th>Prediction:Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10039_14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10039_14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10039_14.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10069_14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10069_14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10069_14.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10077_47.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10077_47.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10077_47.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10087_31.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10087_31.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10087_31.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10103_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10103_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10103_6.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/images/10128_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test/masks/10128_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/UTSW-Glioma-T2W/mini_test_output/10128_8.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient 
Population (BraTS-Africa)</b><br>
Maruf Adewole, Jeffrey D. Rudie, Anu Gbadamosi, Oluyemisi Toyobo, Confidence Raymond, Dong Zhang,<br>
Olubukola Omidiji, Rachel Akinola, Mohammad Abba Suwaid, Adaobi Emegoakor, Nancy Ojo, Kenneth Aguh, <br>
Chinasa Kalaiwo, Gabriel Babatunde, Afolabi Ogunleye, Yewande Gbadamosi, Kator Iorpagher, Evan Calabrese,<br>
Mariam Aboian, Marius Linguraru, Jake Albrecht, Benedikt Wiestler, Florian Kofler, Anastasia Janas,<br>
Dominic LaBella, Anahita Fathi Kzerooni, Hongwei Bran Li, Juan Eugenio Iglesias, Keyvan Farahani, <br>
James Eddy, Timothy Bergquist, Verena Chung, Russell Takeshi Shinohara, Walter Wiggins, Zachary Reitman,<br>
Chunhao Wang, Xinyang Liu, Zhifan Jiang, Ariana Familiar, Koen Van Leemput, Christina Bukas, Maire Piraud,<br>
Gian-Marco Conte, Elaine Johansson, Zeke Meier, Bjoern H Menze, Ujjwal Baid, Spyridon Bakas, Farouk Dako,<br>
Abiodun Fatade, Udunna C Anazodo<br>
<a href="https://arxiv.org/pdf/2305.19369">https://arxiv.org/pdf/2305.19369</a>
<br><br>
<b>2. Advancing Precision: A Comprehensive Review of MRI Segmentation Datasets from BraTS Challenges (2012–2025)</b><br>
Beatrice Bonato, Loris Nanni and Alessandra Bertoldo<br>
<a href="https://www.mdpi.com/1424-8220/25/6/1838">https://www.mdpi.com/1424-8220/25/6/1838</a>
<br><br>
<b>3. Multi-class glioma segmentation on real-world data with missing MRI sequences: comparison of three deep learning algorithms
</b><br>
Hugh G. Pemberton, Jiaming Wu, Ivar Kommers, Domenique M. J. Müller, Yipeng Hu, Olivia Goodkin, <br>
Sjoerd B. Vos, Sotirios Bisdas, Pierre A. Robe, Hilko Ardon, Lorenzo Bello, Marco Rossi, <br>
Tommaso Sciortino, Marco Conti Nibali, Mitchel S. Berger, Shawn L. Hervey-Jumper, Wim Bouwknegt,<br>
Wimar A. Van den Brink, Julia Furtner, Seunggu J. Han, Albert J. S. Idema, Barbara Kiesel,<br>
Georg Widhalm, Alfred Kloet, Michiel Wagemakers, Aeilko H. Zwinderman, Sandro M. Krieg, <br>
Emmanuel Mandonnet, Ferran Prados, Philip de Witt Hamer, Frederik Barkhof & Roelant S. Eijgelaar<br>
<a href="https://www.nature.com/articles/s41598-023-44794-0">
https://www.nature.com/articles/s41598-023-44794-0
</a>
<br><br>
<b>4. Training 3D U-Net for Brain Tumor Segmentation Challenge – Medical Imaging</b><br>
Jaykumaran<br>
<a href="https://learnopencv.com/3d-u-net-brats/">https://learnopencv.com/3d-u-net-brats/</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2023-Subset</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2023-Subset">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2023-Subset
</a>
<br><br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2020</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2020">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2020
</a>
<br><br>
<b>7. TensorFlow-FlexUNet-Image-Segmentation-Brain-Tumor-BraTS2019-HGG-LGG-MRI</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Brain-Tumor-BraTS2019-HGG-LGG-MRI">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Brain-Tumor-BraTS2019-HGG-LGG-MRI
</a>
<br><br>
<b>8. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
