# acoustic_repLearn_dicova2  

#### Filter-bank learning on Coswara Data.
For running the acoustic filter-bank learning experiment for breathing sounds, follow the below steps:  
```
cd exp/breathing_acfb
./submit_job.sh
```
Follow similar steps for running filter-bank learning experiments for speech sounds by changing dir to ```speech_acfb```. 

#### Pretraining Experiments. 
This work utilizes Covid19-Sounds dataset for pretraining of filters followed by optionally fine-tuning on Coswara data. 
For pretraining experiments on breathing sounds, follow the steps:
```
cd exp/breathing_acfb_preTrainedCambridge
./submit_job.sh 
```
