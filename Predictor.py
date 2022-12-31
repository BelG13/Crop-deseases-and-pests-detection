import tensorflow as tf
import pickle as pkl
import numpy as np


class Predictor:

    def __init__(self , modelA , modelB , modelC):

        self.predictor = [modelA , modelB , modelC]

        self.class_names = {
            'potatoes': ['Potato___Early_blight' , 'Potato___healthy' , 'Potato___Late_blight'],

            'peppers' : ['Pepper__bell___Bacterial_spot' , 'Pepper__bell___healthy'],

            'tomatoes' : ['Tomato__Target_Spot' ,
                          'Tomato__Tomato_mosaic_virus' ,
                          'Tomato__Tomato_YellowLeaf__Curl_Virus' ,
                          'Tomato_Bacterial_spot',
                          'Tomato_Early_blight' ,
                          'Tomato_healthy' ,
                          'Tomato_Late_blight' ,
                          'Tomato_Leaf_Mold' ,
                          'Tomato_Septoria_leaf_spot',
                          'Tomato_Spider_mites_Two_spotted_spider_mite' 
                        ]
        }



    def predict(self , image_batch):

        # we store all the value that'll be predicted from the batch 
        # in <preds>

        preds = []


        for img in image_batch:

            # we get the class predicted with the associated probability

            pred_A  = self.predictor[0].predict(img)
            class_a = np.argmax(pred_A)
            pred_A  = pred_A[np.argmax(pred_A)]

            pred_B  = self.predictor[1].predict(img)
            class_b = np.argmax(pred_B)
            pred_B  = pred_B[np.argmax(pred_B)]

            pred_C  = self.predictor[0].predict(img)
            class_c = np.argmax(pred_B)
            pred_C  = pred_C[np.argmax(pred_C)]

            # we store them in arrays

            probas   = np.array([pred_A , pred_B , pred_C])
            classes = np.array([class_a , class_b , class_c])

            # we get the higher probability and give the 
            # associated resulst as prediction

            higher = np.argmax(probas)

            if higher == 0:
                preds.append(self.class_names['potatoes'][classes[0]])

            elif higher == 1:
                preds.append(self.class_names['peppers'][classes[1]])

            elif higher == 2:
                preds.append(self.class_names['tomatoes'][classes[2]])


        # we return either the prediction or the predictions

        if len(preds) == 1:
            return preds[0]

        else:
            return preds




# dataset import
dataset_1 = tf.keras.utils.image_dataset_from_directory(
    directory  = 'PlantVillage/Potatoes',
    batch_size=1
)

dataset_2 = tf.keras.utils.image_dataset_from_directory(
    directory  = 'PlantVillage/peppers',
    batch_size=1
)

dataset_3 = tf.keras.utils.image_dataset_from_directory(
    directory  = 'PlantVillage/Tomatoes',
    batch_size=1
)

model = pkl.load(open('potatoes_model.obj' , 'rb'))

#data = dataset_1.concatenate(dataset_2.concatenate(dataset_3))


# model = Predictor(
#     modelA = pkl.load(open('potatoes_model.obj' , 'rb')),
#     modelB = pkl.load(open('peppers_model.obj' , 'rb')),
#     modelC = pkl.load(open('tomatoes_model.obj' , 'rb'))
# )


# print(data.class_names)

# for image_batch , label_batch in data.take(1):

#     print()





        



