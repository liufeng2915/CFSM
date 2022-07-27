# python 3.7
"""Demo."""

import numpy as np
import torch
import time
import streamlit as st
import SessionState
from itertools import cycle
import cv2
import glob
from utils import load_generator, to_tensor
from torchvision import transforms
preproc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
image_folder = './test_img/'
img_list = glob.glob(image_folder+'*.jpg')
#num_images = 5

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    return load_generator(model_name)


def sample(model, style_dim=10, num=1):
    """Samples latent codes."""
    codes = np.random.normal(0, 1, (num, style_dim))
    return codes


def imageRead(img_list):
    """Samples latent codes."""
    read_idx = np.random.permutation(len(img_list))
    image = cv2.imread(img_list[read_idx[0]])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

#@st.cache(allow_output_mutation=True, show_spinner=False)
def synthesize(model, input_img, code):
    _,syn_img,_ = model(to_tensor(input_img).unsqueeze(0).repeat(code.shape[0],1,1,1), to_tensor(code))
    return syn_img


def main():
    """Main function (loop for StreamLit)."""
    rand_image = imageRead(img_list)
    
    st.set_page_config(layout="wide")
    #st.title('Controllable Face Synthesis')
    st.markdown("<h1 style='text-align: center; color: black;'>Controllable Face Synthesis</h1>", unsafe_allow_html=True)
    st.sidebar.title('Options')
    #reset = st.sidebar.button('Reset All')
    reset_image = st.sidebar.button('Reset Image')
    reset_style = st.sidebar.button('Reset Style')

    
    model_name = st.sidebar.selectbox(
        #'Model to Interpret',
        'Target Dataset',
        ['AgeDB', 'CFP', 'IJB-B', 'IJB-S', 'LFW', 'WiderFace'])
    model = get_model(model_name)

    ## magnitude
    mag_value = st.sidebar.slider(
        'Magnitude',
        value=3.0,
        min_value=0.0,
        max_value=6.0,
        step=0.05)

    num_images = 5

    num_semantics = st.sidebar.number_input(
        'Number of Components', value=5, min_value=1, max_value=10, step=1) 
    max_step = 3.0     
    std_value = st.sidebar.slider(
        'Component STD',
        value=max_step,
        min_value=-max_step,
        max_value=max_step,
        step=0.5)
    #std_value = std_value/3*4
    col0_1, col0_2 = st.columns([0.2,0.8])
    col0_2.header("Target Dataset")
    col02_image_placeholder = col0_2.empty()

    ##
    show_syn_img = np.ones((112, num_semantics*112+(num_semantics-1)*10, 3))*[0.0,255.0,255.0]
    show_img = np.ones((112, (num_images)*112+(num_images-1)*10, 3))*[0.0,255.0,255.0]
    show_target_img = np.ones((112, 5*112+(5-1)*10, 3))*[0.0,255.0,255.0]

    ## 
    col1_1, col1_2 = st.columns([0.2,0.8])
    col1_1.header("Input Face")
    col1_2.header("Synthesized Faces")
    col11_image_placeholder = col1_1.empty()
    col12_image_placeholder = col1_2.empty()

    col2_1, col2_2 = st.columns([0.2,0.8])
    col2_2.header("Learned Bases (1st---->k-th)")
    col22_image_placeholder = col2_2.empty()

    base_codes = sample(model, style_dim=10, num=10)
    state = SessionState.get(model_name=model_name,
                             codes=base_codes,
                             image=rand_image,
                             show_target_img=show_target_img,
                             show_syn_img=show_syn_img,
                             show_img=show_img,
                             )

    if reset_style:
        state.codes = sample(model, style_dim=10, num=num_images)
    if reset_image:
        state.image = imageRead(img_list) 

    ## random styles and input image
    code = state.codes.copy()
    images = state.image.copy()
    col11_image_placeholder.image(cv2.resize(images, [205,205], interpolation=cv2.INTER_AREA)/255.0)
    input_img = preproc(images)#.cuda()
    input_img = input_img.data.cpu().numpy()

    ##
    show_target_img = cv2.imread('./docs/target_dataset_examples/'+model_name+'.jpg')
    show_target_img = cv2.cvtColor(show_target_img, cv2.COLOR_BGR2RGB)
    state.show_target_img = show_target_img
    ## 
    col02_image_placeholder.image(state.show_target_img.copy()/255.0)

    ## generate bases images
    sem_codes = np.zeros((num_semantics, 10))
    for it in range(num_semantics):
        one_hot = np.zeros((1, 10))
        one_hot[:,it] = 1
        sem_codes[it,:] = one_hot * std_value
    sem_syn_img = synthesize(model, input_img, sem_codes)
    sem_syn_img.div_(2).sub_(-0.5).div_(1/255.0)
    sem_syn_img = sem_syn_img.permute(0, 2, 3, 1).data.cpu().numpy()
    #print(sem_syn_img.shape)
    for i in range(sem_syn_img.shape[0]):
        if i == 0:
            start_idx = 0
        else:
            start_idx = start_idx + (112+10)
        show_syn_img[:,start_idx:(start_idx+112),:] = sem_syn_img[i]
    show_syn_img = cv2.resize(show_syn_img, [int(show_syn_img.shape[1]*2),int(show_syn_img.shape[0]*2)], interpolation=cv2.INTER_AREA)
    ## show bases
    state.show_syn_img = show_syn_img
    col22_image_placeholder.image(state.show_syn_img.copy()/255.0)


    ## show random style
    code = code/np.linalg.norm(code, axis=1, keepdims=True)
    code = code*(mag_value/2)
    syn_img = synthesize(model, input_img, code)
    syn_img.div_(2).sub_(-0.5).div_(1/255.0)
    syn_img = syn_img.permute(0, 2, 3, 1).data.cpu().numpy()

    for i in range(num_images):
        if i == 0:
            start_idx = 0
        else:
            start_idx = start_idx + (112+10)
        show_img[:,start_idx:(start_idx+112),:] = syn_img[i]
    show_img = cv2.resize(show_img, [int(show_img.shape[1]*2),int(show_img.shape[0]*2)], interpolation=cv2.INTER_AREA)
    state.show_img=show_img
    col12_image_placeholder.image(state.show_img.copy()/255.0)




if __name__ == '__main__':
    main()
