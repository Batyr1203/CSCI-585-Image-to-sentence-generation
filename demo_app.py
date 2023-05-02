import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
from imageio import imread
from cv2 import resize as imresize
import skimage
import io
import argparse

st.set_page_config(layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search_lstm(encoder, decoder, img, word_map, rev_word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
#    img = imageio.v2.imread(image_inut)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds].long()]
        c = c[prev_word_inds[incomplete_inds].long()]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    
    caption = ' '.join([rev_word_map[s] for s in seq][1:-1])

    return caption, seq, alphas


def caption_image_beam_search_transformer(args, encoder, decoder, img, word_map, device):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = args.beam_size
    Caption_End = False
    vocab_size = len(word_map)
    # Read image and process
#    img = imageio.imread(image_path)
    # img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize((256, 256)))
    # img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(-1)
    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # [1, num_pixels=196, encoder_dim]
    num_pixels = encoder_out.size(1)
    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    if args.decoder_mode == "lstm":
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    elif args.decoder_mode == "transformer":
        k_prev_words = torch.LongTensor([[word_map['<start>']] * 52] * k).to(device)  # (k, 52)


    # Tensor to store top k sequences; now they're just <start>
    seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    if args.decoder_mode == "lstm":
        h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        if args.decoder_mode == "lstm":
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)  # (s, 1, enc_image_size, enc_image_size)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            scores = decoder.fc(h)  # (s, vocab_size)
        elif args.decoder_mode == "transformer":
            cap_len = torch.LongTensor([52]).repeat(k, 1)  # [s, 1]
            scores, _, _, alpha_dict, _ = decoder(encoder_out, k_prev_words, cap_len)
            scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            # choose the last layer, transformer decoder is comosed of a stack of 6 identical layers.
            alpha = alpha_dict["dec_enc_attns"][-1]  # [s, n_heads=8, len_q=52, len_k=196]
            alpha = alpha[:, 0, step-1, :].view(k, 1, enc_image_size, enc_image_size)  # [s, 1, enc_image_size, enc_image_size]

        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds]], dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        # Set aside complete sequences
        if len(complete_inds) > 0:
            Caption_End = True
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        if args.decoder_mode == "lstm":
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        elif args.decoder_mode == "transformer":
            k_prev_words = k_prev_words[incomplete_inds]
            k_prev_words[:, :step + 1] = seqs  # [s, 52]
            # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    assert Caption_End
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = image.resize([14 * 24, 14 * 24], Image.Resampling.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    plt.figure(figsize=(12, 10))
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=20)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    image_pil = Image.open(img_buf)
    return image_pil


@st.cache_resource
def load_lstm():
    print('Loading LSTM')

    model_path = 'checkpoints/lstm_model_checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map_path = 'checkpoints/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

    checkpoint = torch.load(model_path, map_location=str(device))

    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
        
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    return decoder, encoder, word_map, rev_word_map

decoder_lstm, encoder_lstm, word_map, rev_word_map = load_lstm()
    

@st.cache_resource
def load_transformer():
    print('Loading Transformer')

    args = argparse.Namespace(img='example_images/cat.png', 
                            checkpoint='checkpoints/transformer_model_checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', 
                            word_map='checkpoints/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 
                            decoder_mode='transformer', 
                            save_img_dir='./caption', 
                            beam_size=5, 
                            smooth=True)
    
    checkpoint_transformer = torch.load(args.checkpoint, map_location=str(device))
    decoder_transformer = checkpoint_transformer['decoder']
    decoder_transformer = decoder_transformer.to(device)
    decoder_transformer.eval()

    encoder_transformer = checkpoint_transformer['encoder']
    encoder_transformer = encoder_transformer.to(device)
    encoder_transformer.eval()

    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    return args, decoder_transformer, encoder_transformer, word_map, rev_word_map

args, decoder_transformer, encoder_transformer, word_map, rev_word_map = load_transformer()


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return image_data
    else:
        return None
    
def stateful_button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]

    return st.session_state[key]


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    st.title('Image-to-Sentence Generation (CSCI 585 Term Project)')
    st.subheader('By :violet[Aisana Bolatbek], :violet[Batyr Arystanbekov] and :violet[Olzhas Shortanbaiuly]')

    # st.markdown('''
    # <style>
    #     div.stButton button {
    #     background-color: rgb(182, 15, 182);
    #     width: 300px;
    #     height: 40px;
    #     color: white;
    # }
    # </style>''', unsafe_allow_html=True)

    local_css("style.css")

    col1, col2, col3 = st.columns(3)   

    with col1:
        image_data = load_image()
        if image_data is not None:
            image = imageio.v2.imread(image_data)
            image_pil = Image.open(io.BytesIO(image_data))


            if stateful_button('Run LSTM on the image', key='lstm'):
                with col2:
                    st.markdown('# LSTM Model')
                    caption, seq, alphas = caption_image_beam_search_lstm(encoder_lstm, decoder_lstm, image, word_map, rev_word_map)
                    alphas = torch.FloatTensor(alphas)

                    print('predicted caption:', caption)
                    st.markdown(f'## :blue[{caption}]')

                    vis_image = visualize_att(image_pil, seq, alphas, rev_word_map)
                    st.image(vis_image)


            if stateful_button('Run Transformer on the image', key='transformer'):
                with col3:
                    st.markdown('# Transformer Model')
                    with torch.inference_mode():
                        seq, alphas = caption_image_beam_search_transformer(args, encoder_transformer, decoder_transformer, image, word_map, device)
                        alphas = torch.FloatTensor(alphas)
                    caption = ''
                    for s in seq[1:-1]:
                        caption += rev_word_map[s] + ' '

                    print('predicted caption:', caption)
                    st.markdown(f'## :violet[{caption}]')

                    vis_image = visualize_att(image_pil, seq, alphas, rev_word_map)
                    st.image(vis_image)
    

if __name__ == '__main__':
    main()
