# igc
Infrastructure Goal Condition  Reinforce Learner

Reinforcement learning (RL) has indeed shown great success in solving various 
problems by learning optimal decision-making policies. In the context you mentioned, 
the focus is on formulating real-life problems related to cloud infrastructure deployment as a finite 
Markov Decision Process (MDP) and developing an RL agent that can learn to act within this 
environment to achieve a predefined goal.

This proposal describes the end-to-end system. Furthermore, the study examines a specific problem within the 
realm of cloud infrastructure, aiming to identify the optimal strategy for deploying infrastructure from the 
most minimal initial configuration state. Thus, our work cast the entire problem as a Goal condition to 
Reinforce the Learning problem.

# Model Architecture

# Installation

```bash
conda create -n igc python=3.10
conda activate igc
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install 'transformers[torch]'
pip install deepspeed
pip install fairscale
pip install asv
pip install pynvml
pip install 'gym[all]'
pip install loguru
pip install tensorboard
pip install tabulate
pip install pynvml
pip install evaluate
pip install rouge_score
pip install scikit-learn
pip install deepspeed
```

## High level

Before you train, please read because you essentially need to train 4-5 models.  
So it is a bit of a journey. 

At a high level, the training procedure for the language model (LLM) consists of three different steps, 
although the latter two are not necessary if you only want to train the RL agent.

### State Representation Model

The first model focuses on representing the state. Each API response, along with any parameters
that the REST API modifies, can be considered as a small Markov Decision Process (MDP).

Therefore, the state observed by the RL agent corresponds to the GET or HEAR query made 
to a specific REST API.  This step involves defining 
the state representation for the RL agent.

### Data Collection from the Target REST Server 

In this step, the system needs to collect sufficient data from the target system that expose REST API interface. 
My system utilizes tools to interact with the Redfish API. To begin, the first part involves 
gathering data from any Redfish host. The system collects the JSON representations of every API 
supported by the system through a recursive walk.

[Tool used to discover](https://github.com/spyroot/idrac_ctl)

```bash
# tool to collect data from redfish rest system
# note this tool will be integrated directly into the trainer so separate args will first
# collect the data
igc_ctl
```

In my research, I mainly focus on redfish API since the API provides information about how to 
interact with the system, so the hop here is that the agent will learn.  Note that during discover 
phase all json stored in "~/.json_responses"  not that I provide entire dataset
that you don't need build it.

Hence, after discover, all responses are stored in JSON format locally. Now depend on the execution 
mode if "~/.json_responses" present in the system and no local dataset presents the initial 
call to a trainer will rebuild the entire dataset.

[Json dataset](https://github.com/spyroot/igc/blob/main/igc/ds/redfish_dataset.py)

Note that there are several steps involved in this process. In short, all collected data 
is passed to the LLM tokenizer and stored in tensor format. During the discovery phase,
we also collect and store
collection of all API and file names for each response. It is stored as a numpy file 
and consists of mapping API a path to the response that holds the API response 
for GET and HEAD.

* GET store a view for particular API
* HEAD use to figure out what methods each API supports.  (POST/DELETE/PATCH etc)

The action space is concatenation of one hot vector for each API, one hot vector 
for a method and goal, where store separately and never concatenated to action.

In order to understand the need to store the path to the response, let's focus on the 
gym environment that is provided.

As part of the environment, the Rest API server serves as a mock object with 
two options available:

* It can execute a real API call and forward it to the actual Redfish server.

* It can execute a mock request.

Since all the trajectories have been collected, it is possible to simulate any 
GET API request that does not mutate a state by reading the collected 
file and responding offline.

For teaching the agent a specific goal, let's consider a scenario where we want the agent to
perform a valid Redfish POST API request that requires providing some JSON data and mutates a state. 
By default, the mock servers respond to POST/PATCH/DELETE API requests with 
a 404 status code and a JSON error.

However, the environment allows for the registration of handlers by external objects. 
For example, if we have 2000 APIs and we only want to teach the agent to change a password, 
we can register a callback with error and success messages specifically for that goal. 
This allows customization of the responses based on specific goals or objectives.

Let say we train on goal change a password.
We need a mutate a state.  (change password) (POST or PATCH)
Observe a state. (GET)

* You can read JSON and convert to a dictionary.
* register callback that take JSON payload, decode and update internal dict.
* Since we do have initial representation for each API all action that mutate can be offline.
* Hence, you can register callback that will reward an agent if agent changed password.

During the initial dataset build process, it is crucial to make the responses available 
for the Mock server for a read operation. To achieve this, during the construction of the JSON dataset, 
all the responses are compressed and stored along with the dataset. When a regular client needs 
to obtain the dataset, they can download this data from the provided mirrors.

On the client side, once the dataset is obtained, the responses are unpacked and stored 
in a  separate folder within the datasets directory. This dataset becomes a vital component 
of the system as it is used in all the models and algorithms employed in the training process. 
The stored responses serve as a reference for the Mock server, enabling it to mimic real  
API behavior and providing appropriate responses during the RL agent's interactions 
with the environment.

## Tokenizer and Latent representation.

During the dataset creation process, the GPT-2 tokenizer is expanded by default, 
which in turn expands the dimensions of the embeddings. As a result, when 
loading the dataset, the initial GPT-2 model needs to be resized accordingly.

The second step of dataset building focuses on training the Language Model (LLM). 
In this step, there are two main tasks:

* The LLM should be able to extract meaningful latent representations of the state. 
For example, consider the output of the last hidden state is [batch_size, 1023, 768].

* When training the RL agent, consider if we use typically a 3-layer Multi-Layer Perceptron (MLP). 
feed forward network. Hence, we can note that just first linear layer alone requires 1023 * 768 parameters. 

Hence, so far, and this is my current proposal in terms of observation and state representation. 

- First, we train LLM, and in order to do that, we need to teach LLM JSON representation. 
That way I do register special tokens - JSON Object, JSON Array etc , odata.id , target, etc.

To fine-tune the GPT model, a separate dataset is created by inheriting from the main dataset 
and implementing various masking techniques. In this approach, specific REST API requests are 
masked to ensure that the attention mask focuses on the REST API within the original JSON response. 
You can check the code can to understand the details of the masking process. 

The main point here we want to teach LLM keep attention to REST API in responses,
keep attention to all parameters particular API takes,  action in REST API. (i.e reset compute,
change bios etc.)

After completing the training of the LLM, the current approach involves utilizing a separate 
Auto Encoder and 1D convolution pooling.  The goal here reduce the dimensionality 
of the output from the last hidden layer of LLM encoder. 

This step aims to handle scenarios where the API response for a single GET request might consist 
of a very large JSON object, potentially containing thousands of lines. To process such data, 
it needs to be split into smaller chunks and passed through the 
LLM encoder.

Furthermore, it has been found through experimentation that directly passing the
last hidden shape [1023, 768] from the LLM encoder as an observation to the RL agent 
can be complex and challenging. Therefore, the approach of using an autoencoder 
and 1D convolution pooling helps in simplifying the representation and reducing 
the dimensionality, making it more suitable for the RL agent to process effectively.

## Phase 2

We train auto encoder and reduce dimension of LLM encoder output.  Note that if you do many GPU
you probably can collapse this step during first phase and attach auto encoder and train
but it very hard to fit even in 4 GPU system with 24Gb memory



# Instruction

* Training state encoder

```bash
python trainer.py --train llm --num_train_epochs 1000 \
--llm latent --llm_log_level info --log_level info --device cuda:1
```

* Training auto state encoder.

```bash
python trainer.py --train llm --num_train_epochs 1000 --llm encoder --llm_log_level info --log_level info
```

``accelerate
``
```
accelerate launch --config_file /home/spyroot/.cache/huggingface/accelerate/default_config.yaml trainer.py
```

MODEL_NAME=gpt2-xl
PER_DEVICE_TRAIN_BATCH_SIZE=1
HF_PATH=~/projects
NEPOCHS=1
NGPUS=2
NNODES=1
MAX_STEPS=50
OUTPUT_DIR=./output_b${PER_DEVICE_TRAIN_BATCH_SIZE}_g${NGPUS}_$MAX_STEPS


deepspeed --num_gpus=2 run_clm.py \
--deepspeed ../dsconfigs/ds_config_fp16_z2.json\
--model_name_or_path $MODEL_NAME \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--do_train \
--fp16 \
--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
--learning_rate 2e-5 \
--num_train_epochs $NEPOCHS \
--output_dir ${OUTPUT_DIR}_z2 \
--overwrite_output_dir \
--save_steps 0 \
--max_steps $MAX_STEPS \
--save_strategy "no"