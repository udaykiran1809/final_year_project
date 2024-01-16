import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from PIL import Image
from tensorflow.keras.utils import img_to_array
import numpy as np
from keras.layers import Dropout
from keras.models import load_model
import gradio as gr

le = LabelEncoder()

def cortisol_stat(X, y):
    
    X['Gender'] = le.fit_transform(X['Gender'])
    X['Period'] = le.fit_transform(X['Period'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=600, random_state=50)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return rf

def vulnerability_cort(X_1, y_1):
    X_1['Gender'] = le.fit_transform(X_1['Gender'])
    X_1['Period'] = le.fit_transform(X_1['Period'])
    X_1['Status'] = le.fit_transform(X_1['Status'])

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
    rf_1 = RandomForestClassifier(n_estimators=600, random_state=50)
    rf_1.fit(X_train_1, y_train_1)
    y_pred_1 = rf_1.predict(X_test_1)
    accuracy = accuracy_score(y_test_1, y_pred_1)
    print(f"Accuracy: {accuracy}")
    return rf_1

def gender(gen_temp):
    gen_temp = gen_temp.lower()
    if gen_temp == 'male':
        return 1
    else:
        return 0
    
def period(per_temp):
    per_temp = per_temp.lower()
    if per_temp == "early morning":
            return 2
    elif per_temp == "morning":
        return 4
    elif per_temp == "afternoon":
        return 0
    elif per_temp == "evening":
        return 3
    elif per_temp == "before bed":
        return 1
    
def image_preprocess(image):
    image = np.copy(image)
    image = Image.fromarray(image).resize((224, 224))
#     image = image.resize((224, 224))
#     image = img_to_array(image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
    
def model_1(path):
    model = load_model(path, custom_objects={"FixedDropout": FixedDropout})
    return model

def prediction_class(k):
    if k[0] == 0:
        return "Mild"
    elif k[0] ==1:
        return "Moderate"
    elif k[0] == 2:
        return "No_Dr"
    elif k[0] == 3:
        return "Proliferate_DR"
    else:
        return "Severe"
    
def vulnerability_DB(pre):
    if pre == "Proliferate_DR" or pre == "Severe":
        return "Highly Vulnerable"
    elif pre == "Mild" or pre == "Moderate":
        return "less Vulnerable"
    else:
        return "less Vulnerable"
    

def random_forest(age,gen_temp,per_temp,cot, flag):
    R = ""
    gen = gender(gen_temp)
    per = period(per_temp)
    data = {'Age':[age],'Gender':[gen],'Period':[per],'Cortisol(ng/ml)':[cot]}
    df_2 = pd.DataFrame(data)

    K_test = df_2[['Age','Gender','Period','Cortisol(ng/ml)']]
    L_test = rf.predict(K_test)
    for i in L_test:
        if i == 'High':
            K = 0
        elif i == 'Normal':
            K = 2
        else:
            K = 1
            
    df_2['Status'] = K
    M_test = df_2[['Age','Gender','Period','Cortisol(ng/ml)','Status']]
    N_test = rf_1.predict(M_test)
    r=N_test.tolist()
    R = r[0]
    
    if flag:
        tem = True
    else:
        tem = False
    
    if flag and (K == 0 or K == 1):
        
        return f"The person cortisol levels are {i} and further evaluation is required"
        
    else:
        return f"The person cortisol levels are {i} and {R} to Cvd"
    

def Diabetic_retino(image):
    image = image_preprocess(image)
    model = model_1("C:/Users/kiran/Downloads/Diabetic_test.h5")
    predictions = model.predict(image)
    predictions = np.argmax(predictions, axis=1)
    k = predictions.tolist()
    pre = prediction_class(k)
#     temp = vunerability_DB(pre)
    
    return f"The person is {vulnerability_DB(pre)}"

if __name__ == "__main__":

    data = pd.read_csv("C:/Users/kiran/OneDrive - MSFT/Desktop/research/test_1.csv")
    data = shuffle(data,random_state=101)
    X = data[['Age','Gender','Period','Cortisol(ng/ml)']]
    y = data['Status']
    rf = cortisol_stat(X, y)
    X_1 = data[['Age','Gender','Period','Cortisol(ng/ml)','Status']]
    y_1 = data[['vulnerability']]
    rf_1 = vulnerability_cort(X_1,y_1)

    age = gr.inputs.Number(label="Age")
    gen_temp = gr.inputs.Textbox(label="Gender")
    per_temp = gr.inputs.Textbox(label="Period")
    cot = gr.inputs.Number(label="Cortisol(ng/ml)")
    flag = gr.inputs.Checkbox(label="Is diabetic?")

    with gr.Blocks() as UI:
        gr.Markdown("Prediction of Cardiovascular disease using deep learning techniques")
            
        with gr.Tab("Cortisol"):
            
            
            age = gr.inputs.Number(label="Age")
            gen_temp = gr.Textbox(label="Gender")
            per_temp = gr.Textbox(label="Period")
            cot = gr.Number(label="Cortisol(ng/ml)")
            flag = gr.Checkbox(label="Is diabetic?")
            text_output = gr.Textbox(label="Output")
            text_button = gr.Button("Submit")
            
        with gr.Tab("Diabetic Retinopathy"):
            
            image_input = gr.Image()
            image_output = gr.Textbox(label="Output")
            image_button = gr.Button("Predict")
            
        text_button.click(random_forest, 
                        inputs = [age,gen_temp,per_temp,cot,flag], 
                        outputs = text_output)
        
        image_button.click(Diabetic_retino,
                        inputs = image_input, 
                        outputs = image_output)
                
        
    UI.launch(share=True)