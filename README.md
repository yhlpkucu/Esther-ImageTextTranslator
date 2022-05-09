# Esther-ImageTextTranslator

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
COMSE6998_009_2022 Deep Learning System Performance | Final Project

###  🌟 Project Overview



<details open>
<summary>
 <strong>✨ Description</strong>
</summary>
  
<br/>
  
Current translation applications force users to manually input text sentences from a foreign language, which is both inconvenient to input and easily miswritten. Users might give up their translation in the end. We simplify this process by letting users simply input live photos with text, and return real time translation results in any user-specified language. 
  
We use **EAST** as the detection model and **ASTER** as the recognization model. Then we use the **google translation API** to acquire translation result in the target language. After launching this feature on the edge devices, users could do text translation in any natural scenario. 

  
  - 🌪 Goal: Make real-time translation from live photos with text
  
  - 💫 Solution Steps:
    1. Detect Pixels containing words
    2. Recognize words in source language
    3. Translate into Specified language
  
  - 🌊 Value:
    - Conveniently complete image -> text translation in any natural scenarios
    - Beneficial for travelling, business meeting, education, etc.
  
</details>

<br/>

<details >
<summary>
 <strong>🍻 Model</strong>
</summary>
  

 <details open>
 <summary>
  <strong>☕️ Model Architecture</strong>
 </summary>
<img width="505" alt="demo" src="https://user-images.githubusercontent.com/63638608/167324908-2a0f45db-e54e-49d1-8f8b-478175fb3358.png">


 </details>

 <br/>


 <details open>
 <summary>
  <strong>🧊 Detector Model</strong>
 </summary>

<img width="505" alt="demo" src="https://user-images.githubusercontent.com/63638608/167324951-57312e67-993a-4e50-879b-985ac1589b03.png">
  
- We use the **EAST** model as the text detector. It has three part, a feature extractor stem, a feature-merging branch and an output layer. The trick here is to do **concatenation along the channel axis to make the feature map more thick and deep**. In feature extractor stem, it first extract a small part and then larger it until forth of the original size in f1. 
  
- Then in merging branch, the upper layer’s output is unpooled and concat with smaller lower layer output until it is deep enough. Finally it output score map, text boxes, text rotation angle and text quadrangle coordinates. By using well-defined loss, It could examine different size, different direction text.


 </details>
 
 <br/>


 <details open>
 <summary>
  <strong>🍰 Recognizor Model</strong>
 </summary>

<img width="505" alt="demo" src="https://user-images.githubusercontent.com/63638608/167324972-e42b063c-7f47-4f07-9dfb-d5d11728015e.png">

- We use **ASTER model** as the word recognizer. ASTER is the combination of two networks: <u>The Rectification Network</u> and the <u>Text Recognition Network</u>. The Rectification Network first resize the network, and use the localization network to predict control points, Then use them to do Thin-Plate-Spline transformation and generate grids and perform sampling to get the rectified image.
  
- The text recognition network received the rectified input from the rectification network, and use the Seq-to-Seq model to solve the recognition problem. The encoder part convert the feature map to feature sequence, and uses a Bidirectional LSTM to capture the long time dependencies between two directions. The decoder part uses the attention based seq-to-seq to capture output, and use the log-softmax to select bidirectional results with higher score.

  
 </details>

</details>

<br/>


### ⚓️ Repository Directory

- `static`: css, JS and pre-trained model

- `templates`: html

- `tess_dict`: result of text recognition

- `testdetect`: model

- `translation.py`: google translation API

- `app.py`: main program

- `app_withoutTrans.py`: Only return recognition result, not translation result.

- `environment.yml`: Record dependencies


### 🚀 How to run?

1. `conda env create -n <YOUR_ENVIRONMENT_NAME> -f environment.yml`

2. `pip install google-cloud-translate`

 - Note: If you want to use the google-cloud API, please follow [this tutorial](https://cloud.google.com/translate/docs/setup) to set up your environment. 

3. `python app.py`

4. Then you can access the web service through URL you defined 
 
 - Note: Default is http://127.0.0.1:5000


### 🌾 Results & Observations

<details >
<summary>
 <strong>🌼 Detection Model</strong>
</summary>
 
  <img width="505" alt="demo" src="https://user-images.githubusercontent.com/63638608/167321711-24f7fa7b-410c-41f6-bef3-ea789ad0ac8b.png">

  💐 Observations:
 
 - Among all models, using either precision, recall or F1 score as the evaluation metrix, EAST is always the superior than other models.
 
 - In terms of speed, EAST is also the fastest model among all.
 
 - Counter-intuitively, there is no clear relationship between accuracy and speed.
 

</details>

<br/>


<details >
<summary>
 <strong>🌸 Recognition Model</strong>
</summary>
  <img width="505" alt="demo" src="https://user-images.githubusercontent.com/63638608/167321711-24f7fa7b-410c-41f6-bef3-ea789ad0ac8b.png">

  💐 Observations:
 
 - Aster is both the most accurate model, but has the largest throughput on both ICDAR-2013 & ICDAR-2015 datasets.
 
 - Model performance is not stable on different dataset, so there is no 'BEST' model.
 
 - There is no clear relationship between model accuracy and throughput.
 

</details>

<br/>




