Hi

To plug and play: Download this repo into the Desktop and run the bash script with the name of the model you want to train. In order to do so, starting on root

cd Desktop/IoTLLM25/SLMs/

chmod +x ./distilgpt2.sh             <=== this makes the shell script runnable. If you want to train a model that is not this one, then change "./distilgpt2.sh" for "./opt.sh" or "./gpt2.sh"

chmod +x ./venv_and_dependencies.sh  <=== creates a virtual environment with all the needed dependencies

./venv_anddependencies.sh

./distilgpt2.sh
