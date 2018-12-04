# Project Introduction
This project is based on CNN5/DenseNet+BLSTM/LSTM+CTC to realize verification code identification. 
This project is only for training the model, If you need to deploy the model, please move to https://github.com/kerlomz/captcha_platform

# Attention
1. Choose your version:
    This project uses GPU for training by default.
    You can use the CPU version by replacing ```tensorflow-gpu==1.10.0``` in the requirements.txt file with ```tensorflow==1.10.0```
    
# Ready to work
   If you want to use the GPU for training, you must first install CUDA and cuDNN:
   https://www.tensorflow.org/install/install_sources#tested_source_configurations
   Please open the above link to view the version of CUDA and cuDNN corresponding to the current TensorFlow version.

# Start
1. Install the python 3.6 environment (with pip)
2. Install virtualenv ```pip3 install virtualenv```
3. Create a separate virtual environment for the project:
    ```bash
    virtualenv -p /usr/bin/python3 venv # venv is the name of the virtual environment.
    cd venv/ # venv is the name of the virtual environment.
    source bin/activate # to activate the current virtual environment.
    cd captcha_trainer # captcha_trainer is the project path.
    ```
4. ```pip install -r requirements.txt```


# Configuration
1. config.yaml - System Config
    ```yaml
    # - requirement.txt  -  GPU: tensorflow-gpu, CPU: tensorflow
    # - If you use the GPU version, you need to install some additional applications.
    # TrainRegex and TestRegex: Default matching apple_20181010121212.jpg file.
    # - The Default is .*?(?=_.*\.)
    # TrainsPath and TestPath: The local absolute path of your training and testing set.
    # TestSetNum: This is an optional parameter that is used when you want to extract some of the test set
    # - from the training set when you are not preparing the test set separately.
    System:
      DeviceUsage: 0.7
      TrainsPath: 'E:\Task\Trains\YourModelName\'
      TrainRegex: '.*?(?=_)'
      TestPath: 'E:\Task\TestGroup\YourModelName\'
      TestRegex: '.*?(?=_)'
      TestSetNum: 1000
    
    # CNNNetwork: [CNN5, DenseNet]
    # RecurrentNetwork: [BLSTM, LSTM]
    # - The recommended configuration is CNN5+BLSTM / DenseNet+BLSTM
    # HiddenNum: [64, 128, 256]
    # - This parameter indicates the number of nodes used to remember and store past states.
    NeuralNet:
      CNNNetwork: CNN5
      RecurrentNetwork: BLSTM
      HiddenNum: 64
      KeepProb: 0.98
    
    # SavedSteps: A Session.run() execution is called a Epochs,
    # - Used to save training progress, Default value is 100.
    # ValidationSteps: Used to calculate accuracy, Default value is 100.
    # TestNum: The number of samples for each test batch.
    # - A test for every saved steps.
    # EndAcc: Finish the training when the accuracy reaches [EndAcc*100]%.
    # EndEpochs: Finish the training when the epoch is greater than the defined epoch.
    Trains:
      SavedSteps: 100
      ValidationSteps: 500
      EndAcc: 0.975
      EndEpochs: 1
      BatchSize: 64
      TestBatchSize: 400
      LearningRate: 0.01
      DecayRate: 0.98
      DecaySteps: 10000
    ```
    There are several common examples of TrainRegex:
    i. apple_20181010121212.jpg
    ```
    .*?(?=_.*\.)
    ```
    ii apple.png
    ```
    .*?(?=\.)
    ```
    
1. model.yaml  - Model Config
    ```yaml
    # Sites: A bindable parameter used to select a model. 
    # - If this parameter is defined, 
    # - it can be identified by using the model_site parameter 
    # - to identify a model that is inconsistent with the actual size of the current model.
    # ModelName: Corresponding to the model file in the model directory,
    # - such as YourModelName.pb, fill in YourModelName here.
    # ModelType: This parameter is also used to locate the model. 
    # - The difference from the sites is that if there is no corresponding site, 
    # - the size will be used to assign the model. 
    # - If a model of the corresponding size and corresponding to the ModelType is not found, 
    # - the model belonging to the category is preferentially selected.
    # CharSet: Provides a default optional built-in solution:
    # - [ALPHANUMERIC, ALPHANUMERIC_LOWER, ALPHANUMERIC_UPPER,
    # -- NUMERIC, ALPHABET_LOWER, ALPHABET_UPPER, ALPHABET]
    # - Or you can use your own customized character set like: ['a', '1', '2'].
    # CharExclude: CharExclude should be a list, like: ['a', '1', '2']
    # - which is convenient for users to freely combine character sets.
    # - If you don't want to manually define the character set manually,
    # - you can choose a built-in character set
    # - and set the characters to be excluded by CharExclude parameter.
    Model:
      Sites: []
      ModelName: YourModelName
      ModelType: 150x50
      CharSet: ALPHANUMERIC_LOWER
      CharExclude: []
      CharReplace: {}
      ImageWidth: 150
      ImageHeight: 50
    
    # Binaryzation: [-1: Off, >0 and < 255: On].
    # Smoothing: [-1: Off, >0: On].
    # Blur: [-1: Off, >0: On].
    Pretreatment:
      Binaryzation: -1
      Smoothing: -1
      Blur: -1
    ```
# Tools
1. Pretreatment Previewer
    ```python -m tools.preview```
2. Navigator (Currently only supports character set recommendations)
    ```python -m tools.navigator```
4. PyInstaller Package
    ```
    pip install pyinstaller
    python -m tools.package
    ```
# Run
1. ```python trains.py```

# License
This project use SATA License (Star And Thank Author License), so you have to star this project before using. Read the license carefully.

# Introduction
https://www.jianshu.com/p/80ef04b16efc
