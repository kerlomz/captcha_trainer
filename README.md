# Project Introduction
This project is based on CNN+LSTM+CTC to realize verification code identification. 
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
    cd captcha_platform # captcha_platform is the project path.
    ```
4. ```pip install -r requirements.txt```


# Configuration
1. config.yaml - System Config
    ```yaml
    # Device: [gpu:0, cpu:0] The default device is GPU.
    # - requirement.txt  -  GPU: tensorflow-gpu, CPU: tensorflow
    # - If you use the GPU version, you need to install some additional applications.
    # TrainRegex and TestRegex: Default matching apple_20181010121212.jpg file.
    # TrainsPath and TestPath: The local path of your training and testing set.
    System:
      NeuralNet: 'CNN+LSTM+CTC'
      DeviceUsage: 0.7
      TrainsPath: 'E:\Task\Trains\YourModelName'
      TrainRegex: '.*?(?=_.*\.)'
      TestPath: 'E:\Task\TestGroup\YourModelName'
      TestRegex: '.*?(?=_.*\.)'
    
    # SavedStep: A Session.run() execution is called a Step,
    # - Used to save training progress, Default value is 100.
    # TestNum: The number of samples for each test batch.
    # - A test for every saved steps.
    # CompileAcc: When the accuracy reaches the set threshold,
    # - the model will be compiled together each time it is archived.
    # - Available for specific usage scenarios.
    # EndAcc: Finish the training when the accuracy reaches [EndAcc*100]%.
    # EndStep: Finish the training when the step is greater than the [-1: Off, EndStep >0: On] step.
    # LearningRate: Find the fastest relationship between the loss decline and the learning rate.
    Trains:
      SavedStep: 100
      TestNum: 500
      CompileAcc: 0.8
      EndAcc: 0.95
      EndStep: -1
      LearningRate: 0.001
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
    # ModelName: Corresponding to the model file in the model directory,
    # - such as YourModelName.pb, fill in YourModelName here.
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
      ModelName: YourModelName
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
3. Quantize(Deleted)
    ```python -m tools.quantize --input=***.pb --output=***.pb```
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
https://www.jianshu.com/p/b1a5427db6e2
