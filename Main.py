import json


# Given a JSON object, it return a dictionary where every single
# images is associated with its captions
##### {1: ['ann1' , 'ann2'], 2: ['ann3', 'ann4']}
# The 'label' parameter allow to do it on a particular section of the images that can be:
##### 'train'   'val'   'test'
def images_captions(data, label):
    images_captions = dict()
    for entry in data['images']:
        if(entry['label'] == label):
            images_captions[entry['img_id']] = []
            for caption in entry['captions']:
                images_captions[entry['img_id']].append(caption)
    return images_captions

def images_filenames(data, label):
    images_filenames = dict()
    for entry in data['images']:
        if(entry['label'] == label):
            images_filenames[entry['img_id']] = entry['img_file_name']
    return images_filenames

##############################
### 1. Prepare the dataset ###
##############################
# Loading the captions
data = json.load(open('data_structure.json'))

# Load the images and extract the captions
images = 'PATH_TO_IMAGES'

train_captions = images_captions(data, "train")
train_images= images_filenames(data, "train")

val_captions = images_captions(data, "val")
val_images= images_filenames(data, "val")

test_captions = images_captions(data, "test")
test_images = images_filenames(data, "test")

##############################
### 2. CAPTIONS PROCESSING ###
##############################
import numpy as np
from nltk import word_tokenize
from collections import Counter

def vocabulary_counter(sentences):
        words_list = []
        for sentence in sentences:
            for word in sentence:
                words_list.append(word)
        words_list = list(dict.fromkeys(words_list))
        return words_list

def captions_preproccessing(train_captions):
    # Collecting all the captions about the training images in a single array
    ##### ["annotation example 1.1", "annotation example 1.2"]
    sentences = []
    for entry in train_captions:
        for caption in train_captions[entry]:
            sentences.append(caption)

    # Lower-case the sentence, tokenize them and add <SOS> and <EOS> tokens (Start - End)
    ##### [['<SOS>', 'annotation', 'example', '1.1', '<EOS>'], ['<SOS>', 'annotation', 'example', '1.2', '<EOS>']]
    sentences = [["<SOS>"] + word_tokenize(sentence.lower()) + ["<EOS>"] for sentence in sentences]

    # Create the vocabulary. Note that we add an <UNK> token to represent words not in our vocabulary.
    vocabulary_words = vocabulary_counter(sentences)
    vocabularySize = len(vocabulary_words) + 2

    # Count the frequency of each single word within the sentences and order them based on it.
    word_counts = Counter([word for sentence in sentences for word in sentence])
    vocabulary = ["<UNK>"] + [e[0] for e in word_counts.most_common(vocabularySize-1)]

    # Assign an index to every single word within the vocabulary
    ##### {'<UNK>': 0, '<SOS>': 1, 'annotation': 2, 'example': 3, '<EOS>': 4, '1.1': 5, '1.2': 6}
    word2index = {word:index for index,word in enumerate(vocabulary)}

    # The following Numpy function allows to create a one_hot_embeddings about the words within the vocabulary
    ##### [[1. 0. 0. 0. 0. 0. 0. 0.], [0. 1. 0. 0. 0. 0. 0. 0.], ... ]
    one_hot_embeddings = np.eye(vocabularySize)

    # Define the max sequence length to be the longest sentence in the training data. 
    maxSequenceLength = max([len(sentence) for sentence in sentences])
    return one_hot_embeddings, maxSequenceLength

one_hot_embeddings, maxSequenceLength = captions_preproccessing(train_captions)

##############################
##### 3. CLASSIFIER MODEL ####
##############################
from keras.applications.vgg16 import VGG16

# Initializate the classifier model, in this case pretrained VGG16
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))


#################################################
##### 4. ENCODING THE IMAGES USING THE MODEL ####
#################################################
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from keras.preprocessing import image

# load an image from file
def load_image(filename):
    x = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Given a list of images, the associated list of captions and a classification model
# It will save a file (under the format of a Dict()) representing the 
# features represented as numpy array.
def images_to_array(images_list, captions_list, vgg_model, label):
        features = {}
        for idx in captions_list:
            # Load/preprocess the image.
            img_path = images_list[idx]
            print(img_path)
            img = load_image(img_path)
            # Run through the convolutional layers and resize the output.
            output = vgg_model.predict(img)
            features[idx] = output

        # For simplicity, we convert this to a numpy array and save the result to a file.
        np.save(open(label + '_features', 'wb+'), features)

# Convert the input images into numpy array and save it
images_to_array(train_images, train_captions, vgg_model, "train")
train_images_features = np.load(open('train_features', 'rb'))

