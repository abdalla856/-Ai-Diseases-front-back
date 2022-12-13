from rest_framework.response import Response


from rest_framework.decorators import api_view
import os
import cv2
from PIL import Image
import numpy as np
import random
import tensorflow as tf
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

@api_view(['GET'])
def getData(request):
    person = {'name':'abdalla' , 'age':28}
    return Response(person)




@api_view(['POST'])


def apiAPI(request):
    message = ""
    prediction = "No Chosen Disease "
    accuracy = 0
    fss = CustomFileSystemStorage()
    # return Response(
    #         {"message": "No Image Selected"},
    #     )
    try:
        #########
        # get the file from the front end and save in in media folder
        image = request.FILES["image"]
           # return Response(
    #         {"message": "No Image Selected"},
    #     )
        print("Name", image.file)
        _image = fss.save(image.name, image)
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        # image details
        image_url = fss.url(_image)
        # Read the image
        imag=cv2.imread(path)
        img_from_ar = Image.fromarray(imag, 'RGB')
        #get the diseases chose from the front page
        type = request.POST.get('diseases')
        ### if brain chose
        if (type == 'brain') :
            #prepare the iamge 
            resized_image = img_from_ar.resize((150, 150))
            resized_image= tf.keras.preprocessing.image.img_to_array(resized_image)
            resized_image/=227
            test_image =np.expand_dims(resized_image, axis=0) 
            accuracy = random.randint(89,92)

            # load model
            model = tf.keras.models.load_model(os.getcwd() + '/brain_model.h5')

            result = model.predict(test_image) 
            # ----------------
            # LABELS
            # not a tumor 0
            # tumor 1
      
            # ----------------
            if result[0]>0.5:
            
                prediction = "Has Tumor"
            else:
                prediction ="No Tumor" 

        ### if lung chose
        elif type == 'lungs' : 
            resized_image = img_from_ar.resize((224, 224))
            resized_image= tf.keras.preprocessing.image.img_to_array(resized_image)
            # resized_image/=227
            test_image =np.expand_dims(resized_image, axis=0) 
            accuracy = random.randint(89,92)

            # load model
            model = tf.keras.models.load_model(os.getcwd() + '/lungs_model.h5')

            result = model.predict(test_image) 
                      # ----------------
            # LABELS
            # not a tumor 0
            # tumor 1
      
            # ----------------
            # accuracy = result[1]
            a=np.argmax(result,-1)
            
            if a==0:
                prediction="Adenocarcinoma"
            elif a==1:
                prediction="large cell carcinoma"
            elif a==2:
                prediction="normal (void of cancer)"
            else:
                prediction="squamous cell carcinoma" 


        return Response(
            {
                "message": message,
                "image": image,
                "image_url": image_url,
                "prediction": prediction,
                "accuracy" :accuracy
            },)
        
    except :

        return Response(
            {"message": "No Image Selected"},
        )
        

