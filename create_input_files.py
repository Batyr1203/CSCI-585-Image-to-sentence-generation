from utils import create_input_files


if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/workspace/datasets/COCO_2014/caption_datasets/dataset_coco.json',
                       image_folder='/workspace/datasets/COCO_2014/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/workspace/datasets/COCO_2014/out/',
                       max_len=50)

# if __name__ == '__main__':
#     # Create input files (along with word map)
#     create_input_files(dataset='flickr8k',
#                        karpathy_json_path='/workspace/datasets/Flickr8k/dataset_flickr8k.json',
#                        image_folder='/workspace/datasets/Flickr8k/Flicker8k_Dataset/',
#                        captions_per_image=5,
#                        min_word_freq=5,
#                        output_folder='/workspace/datasets/Flickr8k/out/',
#                        max_len=50)
