# SCAware
SCAware is an application that predicts sudden cardiac arrest in real-time and alerts users with actionable steps, enabling immediate response to mitigate life-threatening outcomes.

## Use:
Get your Cerebras API Key and add it to enviornment variables using the command prompt
```bash
setx CEREBRAS_API_KEY "YOUR_API_KEY"
```
Check if the API Key is set properly using the command prompt. 
```bash
echo %CEREBRAS_API_KEY%
```
*Note: You may need to restart your machine for changes to take effect.*

Clone the repo and go into the directory
```
git clone https://github.com/SameerDalal/SCAware.git
cd SCAware
```

Create a virtual environment and install dependencies
```
python -m venv env

.\env\Scripts\activate

pip install -r requirements.txt
```

Download the test data:
```
gdown --folder "https://drive.google.com/drive/folders/1eMPxIR26j12avpSAWSJ9vLiYd6nZMukg" -O "./data/test_data/"
```
Download the model:
```
gdown --folder "https://drive.google.com/drive/folders/1M9-qE_Eo3BeuYNICYqQYNOqhMmMpFWau?usp=sharing" -O "./models/model_3/keras/"
```

Run the application with the following command. Choose between simulating normal ECG or SCA ECG
```
streamlit run home.py -- --data_type <data_type>
```
### Example:
To simulate a SCA ECG:
```
streamlit run home.py -- --data_type sca
```
To simulate a normal ECG:
```
streamlit run home.py -- --data_type normal
```



# Citations:
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

Martin-Yebra, A., Martínez, J. P., & Laguna, P. (2024). MUSIC (Sudden Cardiac Death in Chronic Heart Failure) (version 1.0.0). PhysioNet. https://doi.org/10.13026/cec2-9w70.

Mayo Foundation for Medical Education and Research. (2024, December 7). Sudden cardiac arrest. Mayo Clinic. https://www.mayoclinic.org/diseases-conditions/sudden-cardiac-arrest/symptoms-causes/syc-20350634