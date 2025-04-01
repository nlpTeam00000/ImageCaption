import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

    
# @st.cache_resource
# def load_cap_model():
#     return load_model("best_model.h5")

lstm_model = load_model("best_model.h5")
print(type(lstm_model))


def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)
    
_tokenizer = load_tokenizer()

max_length = 35


def idx_to_word(integer, _tokenizer):
    for word, index in _tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def extract_features(img_path):
    # Load VGG16 model pre-trained on ImageNet, remove the top layer
    model = VGG16(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Last FC layer before softmax

    # Load image and preprocess it
    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 requires (224,224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)
    img_array = preprocess_input(img_array)  # Normalize like VGG16 expects

    # Extract features
    features = model.predict(img_array)  # Output shape (1, 4096)
    return features


def predict_caption(model, image_features, _tokenizer, max_length):
    in_text = "startseq"

    for _ in range(max_length):
        # Convert text to sequence
        sequence = _tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Ensure correct input shape
        image_features = np.array(image_features).reshape(1, 4096)  # Ensure (1, 4096)
        sequence = np.array(sequence).reshape(1, max_length)  # Ensure (1, max_length)

        # Predict next word
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, _tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    temp_path = "temp_path.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    print(temp_path)

    st.cache_data.clear()
    features = extract_features(temp_path)
    print("Extracted Features:", features)

    caption = predict_caption(lstm_model, features, _tokenizer, max_length)
    print(caption)
    st.subheader("Generated Caption")
    st.write(caption)

  
# def main(): 
#     st.title("Income Predictor")
#     html_temp = """
#     <div style="background:#025246 ;padding:10px">
#     <h2 style="color:white;text-align:center;">Income Prediction App </h2>
#     </div>
#     """
#     st.markdown(html_temp, unsafe_allow_html = True)
    
#     age = st.text_input("Age","0") 
#     workclass = st.selectbox("Working Class", ["Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay"]) 
#     education = st.selectbox("Education",["10th","11th","12th","1st-4th","5th-6th","7th-8th","9th","Assoc-acdm","Assoc-voc","Bachelors","Doctorate","HS-grad","Masters","Preschool","Prof-school","Some-college"]) 
#     marital_status = st.selectbox("Marital Status",["Divorced","Married-AF-spouse","Married-civ-spouse","Married-spouse-absent","Never-married","Separated","Widowed"]) 
#     occupation = st.selectbox("Occupation",["Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-serv","Prof-specialty","Protective-serv","Sales","Tech-support","Transport-moving"]) 
#     relationship = st.selectbox("Relationship",["Husband","Not-in-family","Other-relative","Own-child","Unmarried","Wife"]) 
#     race = st.selectbox("Race",["Amer Indian Eskimo","Asian Pac Islander","Black","Other","White"]) 
#     gender = st.selectbox("Gender",["Female","Male"]) 
#     capital_gain = st.text_input("Capital Gain","0") 
#     capital_loss = st.text_input("Capital Loss","0") 
#     hours_per_week = st.text_input("Hours per week","0") 
#     nativecountry = st.selectbox("Native Country",["Cambodia","Canada","China","Columbia","Cuba","Dominican Republic","Ecuador","El Salvadorr","England","France","Germany","Greece","Guatemala","Haiti","Netherlands","Honduras","HongKong","Hungary","India","Iran","Ireland","Italy","Jamaica","Japan","Laos","Mexico","Nicaragua","Outlying-US(Guam-USVI-etc)","Peru","Philippines","Poland","Portugal","Puerto-Rico","Scotland","South","Taiwan","Thailand","Trinadad&Tobago","United States","Vietnam","Yugoslavia"]) 
    
#     if st.button("Predict"): 
#         features = [[age,workclass,education,marital_status,occupation,relationship,race,gender,capital_gain,capital_loss,hours_per_week,nativecountry]]
#         data = {'age': int(age), 'workclass': workclass, 'education': education, 'maritalstatus': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capitalgain': int(capital_gain), 'capitalloss': int(capital_loss), 'hoursperweek': int(hours_per_week), 'nativecountry': nativecountry}
#         print(data)
#         df=pd.DataFrame([list(data.values())], columns=['age','workclass','education','maritalstatus','occupation','relationship','race','gender','capitalgain','capitalloss','hoursperweek','nativecountry'])
                
#         category_col =['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
#         for cat in encoder_dict:
#             for col in df.columns:
#                 le = preprocessing.LabelEncoder()
#                 if cat == col:
#                     le.classes_ = encoder_dict[cat]
#                     for unique_item in df[col].unique():
#                         if unique_item not in le.classes_:
#                             df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
#                     df[col] = le.transform(df[col])
            
#         features_list = df.values.tolist()      
#         prediction = model.predict(features_list)
    
#         output = int(prediction[0])
#         if output == 1:
#             text = ">50K"
#         else:
#             text = "<=50K"

#         st.success('Employee Income is {}'.format(text))
      
# if __name__=='__main__': 
#     main()