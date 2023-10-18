# Quinten Best Stocks Prediction
In the scope of this project we aimed to develop an ML-based assistance system for determining the best set of stocks to
trade.

## Project structure
* `bash` - The directory to store bash scripts;
* `data` - The directory to store the data;
* `docs` - The directory with the documentation files;
* `models` - The directory to store pre-trained models' weights;
* `notebooks` - The directory to store ipynb notebooks;
* `references` - The directory with the explanatory materials, helper files, etc.
* `src` - The directory with the source code of the project.

## Setup
```shell
pip install -r requirements.txt
```


To run training, refer to src/models/train.py file.

Command line to run training:
``` python src/models/train.py --training_start "2016/01/31" --training_end "2018/01/31" --validation_start "2018/01/31" --validation_end "2018/06/30" --model_name "model1"```

The training parameters for dates may be modified to run training on different periods. Pay attention to have the training periods matching with the ones on the dataset. Possible values are between 2017/03/31 to 2022/01/31. 

The data only exists for 4 dates on each year. 01/31, 03/31, 06/30, 09/30. Any other dates than the ones writtten will yield erros on training.


