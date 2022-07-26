# Controllable Face Synthesis Model (CFSM)

<img src="docs/teaser.png" width="700px" />

Given an input face image, our CFSM enables precise control of the direction and magnitude of the targeted styles in the generated images. The latent style has both the direction and the magnitude, where the direction linearly combines the learned bases to control the type of style, while the magnitude controls the degree of style. 

## Training

* Please download the released ArcFace model [here](https://drive.google.com/file/d/1GZjPokFv5zXIxol3eZnuByJ0iwDv8inQ/view?usp=sharing) and place it in "/id_weights".
* Prepare the source and target data, and place in "/data"
* Train the model:

```bash
python train.py \
--source_img_path /dataset/v2_39m_source \
--source_list data_list/source_list.txt \
--target_img_path /dataset/widerface \
--target_list data_list/target_list_wf_12k.txt \
--batch_size 32 \
--model_name WiderFace12K \
--n_epochs 5  \
--lambda_identity 8.0   

python train.py \
--source_img_path /dataset/v2_39m_source \
--source_list data_list/source_list.txt \
--target_img_path /dataset/widerface \
--target_list data_list/target_list_wf_12k.txt \
--batch_size 32 \
--model_name WiderFace12K \
--n_epochs 8  \
--epoch 4 \
--lambda_identity 5.0   

python train.py \
--source_img_path /dataset/v2_39m_source \
--source_list data_list/source_list.txt \
--target_img_path /dataset/widerface \
--target_list data_list/target_list_wf_12k.txt \
--batch_size 32 \
--model_name WiderFace12K \
--n_epochs 20  \
--epoch 7 \
--lambda_identity 2.0   
```
