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
import pickle

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
            img = load_image(img_path)
            # Run through the convolutional layers and resize the output.
            output = vgg_model.predict(img)
            features[idx] = output

        # For simplicity, we convert this to a numpy array and save the result to a file.
        pickle_out = open(label + "_features.pickle","wb")
        pickle.dump(features, pickle_out)
        pickle_out.close()

# Convert the input images into numpy array and save it
images_to_array(train_images, train_captions, vgg_model, "train")
pickle_in = open("train_features.pickle","rb")
train_images_features = pickle.load(pickle_in)

#################################################
##### 5. ENCODING THE DATA USING THE MODEL   ####
#################################################

def data_values(data, label):
    data_values = dict()
    for entry in data['images']:
        if(entry['label'] == label):
            data_values[entry['img_id']] = entry['data']
    return data_values

train_data = data_values(data, "train")
val_data = data_values(data, "val")
test_data = data_values(data, "test")


####################################################### ???????????
####################################################### ???????????
####################################################### ???????????

# train_images              #   {1: "FILENAME1.jpeg"}
# train_images_features     #   {1: array([[.....]])}
# train_captions            #   {1: ['ann1' , 'ann2']}
# train_data                #   {1: [1,2,3,4,5,6,7,8,9,10]}

#################################################
##### 6. DEFINE THE MODEL STRUCTURE          ####
#################################################

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
import numpy as np

EMBEDDING_DIM = 128

# Model structure: https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
def create_model():

    # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(4096,), name='main_input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 129-dimensional vectors.
    x = Embedding(output_dim=128, input_dim=4096, input_length=4096)(main_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(32)(x)

    # Change the 4 with the vocabulary size.
    auxiliary_output = Dense(4, activation='sigmoid', name='aux_output')(lstm_out)

    # Change the 10 with the length of data
    auxiliary_input = Input(shape=(10,), name='aux_input')
    x = concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(4, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

    model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})



    ################## !?!?!? ############
    
    main_data = np.array([[1]*4096])                    # IMAGES FEATURES ARRAY
    aux_data = np.array([[1]*10])                  # DATA ARRAY
    main_label = np.array([[0, 0, 0, 1]])         # CAPTIONS
    out_label = np.array([[0, 0, 0, 1]])          # CAPTIONS

    # And trained it via:
    model.fit({'main_input': main_data, 'aux_input': aux_data},
           {'main_output': main_label, 'aux_output': out_label},
            epochs=5, batch_size=1)

    return ""

create_model()