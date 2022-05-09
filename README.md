# Esther-ImageTextTranslator

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
COMSE6998_009_2022 Deep Learning System Performance | Final Project

###  Project Overview



<details open>
<summary>
 <strong>Description</strong>
</summary>
  
<br/>
  
Current translation applications force users to manually input text sentences from a foreign language, which is both inconvenient to input and easily miswritten. Users might give up their translation in the end. We simplify this process by letting users simply input live photos with text, and return real time translation results in any user-specified language. 
  
We use **EAST** as the detection model and **ASTER** as the recognization model. Then we use the **google translation API** to acquire translation result in the target language. After launching this feature on the edge devices, users could do text translation in any natural scenario. 

  
  - Goal: Make real-time translation from live photos with text
  
  - Solution Steps:
    1. Detect Pixels containing words
    2. Recognize words in source language
    3. Translate into Specified language
  
  - Value:
    - Conveniently complete image -> text translation in any natural scenarios
    - Beneficial for travelling, business meeting, education, etc.
  
</details>

<br/>

<details >
<summary>
 <strong>Model</strong>
</summary>
  

</details>

<br/>


### Repository Directory

- `static`: css, JS and pre-trained model

- `templates`: html

- `tess_dict`: result of text recognition

- `testdetect`: model

- `translation.py`: google translation API

- `app.py`: main program

- `app_withoutTrans.py`: Only return recognition result, not translation result.

- `environment.yml`: Record dependencies


### How to run?

1. `conda env create -n <YOUR_ENVIRONMENT_NAME> -f environment.yml`

2. `pip install google-cloud-translate`

3. `python app.py`

4. Then you can access the web service through URL you defined 
 
 - Note: Default is http://127.0.0.1:5000


### Results & Observations

![image](https://user-images.githubusercontent.com/63638608/167321697-162bb75d-cfe9-4178-a625-f6fd90d1d5db.png)

![image](https://user-images.githubusercontent.com/63638608/167321711-24f7fa7b-410c-41f6-bef3-ea789ad0ac8b.png)


