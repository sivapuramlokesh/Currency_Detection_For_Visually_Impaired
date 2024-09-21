from flask import Flask, redirect, url_for, request, render_template, jsonify
import glob
import cv2
import torch
from PIL import Image
from io import BytesIO
from gtts import gTTS
import pyglet
import sys, os, time

# strings at index 0 is not used, it
# is to make array indexing simple
one = [ "", "one ", "two ", "three ", "four ",
        "five ", "six ", "seven ", "eight ",
        "nine ", "ten ", "eleven ", "twelve ",
        "thirteen ", "fourteen ", "fifteen ",
        "sixteen ", "seventeen ", "eighteen ",
        "nineteen "];
 
# strings at index 0 and 1 are not used,
# they is to make array indexing simple
ten = [ "", "", "twenty ", "thirty ", "forty ",
        "fifty ", "sixty ", "seventy ", "eighty ",
        "ninety "];
        
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Build paths inside the project 
print(BASE_DIR)
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['jpg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

  
def get_camera():

    cam = cv2.VideoCapture(0)
    result, image = cam.read()
  
    # If image will detected without any error, 
    # show result
    if result:
  
        # showing result, it take frame name and image 
        # output
        cv2.imshow("GeeksForGeeks", image)
  
        # saving image in local storage
        cv2.imwrite("cam.jpg", image)
  
        # If keyboard interrupt occurs, destroy image 
        # window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  
        # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")
        
    return
    
def speak(say):
    """Writes text to a mp3 file and play the sound in english.

    Args:
      say:any text to convert

    """
    ts = gTTS(text= say, lang='en')
    tsname=("path/name.mp3")
    ts.save(tsname)

    music = pyglet.media.load(tsname, streaming = False)
    music.play()

    time.sleep(music.duration)
    os.remove(tsname)
    return

class CurrencyNotesDetection:
  
    def __init__(self, model_name):        
      
        self.model = self.load_model(model_name)     
        self.classes = self.model.names      
       
    def load_model(self, model_name):        
        
        model = torch.hub.load('./yolov5', 'custom', path=model_name, source='local')  # local repo
        
        return model
        
    def numToWords(self,n, s):
 
        str = ""
        
        # if n is more than 19, divide it
        if (n > 19):
            str += ten[n // 10] + one[n % 10]
        else:
            str += one[n]
    
        # if n is non-zero
        if(n != 0):
            str += s
    
        return str

    def convertToWords(self,n):
        # stores word representation of given
        # number n
        out = ""

        # handles digits at ten millions and
        # hundred millions places (if any)
        out += self.numToWords((n // 10000000),"crore ")

        # handles digits at hundred thousands
        # and one millions places (if any)
        out += self.numToWords(((n // 100000) % 100),"lakh ")

        # handles digits at thousands and tens
        # thousands places (if any)
        out += self.numToWords(((n // 1000) % 100),"thousand ")

        # handles digit at hundreds places (if any)
        out += self.numToWords(((n // 100) % 10),"hundred ")

        if (n > 100 and n % 100):
            out += "and "

        # handles digits at ones and tens
        # places (if any)
        out += self.numToWords((n % 100), "")

        return out

    def get_text(self,labelCnt):
        text = "Image contains"
        noOfLabels,counter = len(labelCnt),0
        for k,v in labelCnt.items():
            text += " {}{} {} ".format(self.convertToWords(v),k,"Notes" if v>1 else "Note")
            if(counter != noOfLabels-1):
                text += 'and'
            counter += 1

        return text


    def get_detected_image(self,img):
        # Images
        imgs = [img]  # batched list of images

        # Inference
        results = self.model(imgs, size=416)  # includes NMS

        # Results
        results.print()  # print results to screen
        # results.show()  # display results
        # results.save()  # save as results1.jpg, results2.jpg... etc. in runs directory
        #print(results)  # models.common.Detections object, used for debugging

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        n = len(labels)
        labelCnt = {}
        for i in range(n):
            classLabel = self.classes[int(labels[i])]
            row = cord[i]
            # row[4] is conf score
            #print("{} is detected with {} probability.".format(classLabel, row[4]))
            if classLabel in labelCnt:
                labelCnt[classLabel] += 1
            else:
                labelCnt[classLabel] = 1

        text = self.get_text(labelCnt)
        print("------------------------------------")
        print(text)
        print("------------------------------------")
        #speak(text)        
        results.imgs
        results.render()  # updates results.imgs with boxes and labels, returns nothing

        #for testing, display results using opencv
        
        for img in results.imgs:
            cv2.imwrite("static/out.jpg", img)
            #cv2.imshow("YoloV5 Detection", cv2.resize(img, (416, 416))[:, :, ::-1])
            #cv2.imshow("YoloV5 Detection", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        
        
        return results.imgs[0],text

    

def run_model(img):    
  
    obj = CurrencyNotesDetection(model_name='./yolov5/weights/best.pt')
    detected_labels_text = ""
    detected_img,detected_labels_text = obj.get_detected_image(img)
    speak(detected_labels_text)
    
    return detected_img, detected_labels_text,"after.html"  
            

    
####################################################################################################   
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict2',methods=['GET','POST'])
def predict2():
    print("Entered")
    
    print("Entered here")
    file = request.files['files'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print("@@ Predicting class......")
    #print(BASE_DIR+'\out.jpg')
    
    pred1 ,pred, output_page = run_model(file_path)
              
    return render_template(output_page, pred_output1 = pred1,pred_output = pred, img_src = '/static/out.jpg')

   
if __name__ == '__main__':
    app.run(debug=False)