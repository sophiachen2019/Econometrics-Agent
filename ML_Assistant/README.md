# MetaGPT Installation and Configuration Guide

## 1. Clone and Install

First, clone the repository to your local machine and install the necessary dependencies:

```bash
https://github.com/FromCSUZhou/ML_Assistant.git
cd ./ML_Assistant
pip install -e .
```

## 2. Edit Configuration

Update the configuration file located at ~/config/config2.yaml according to the example and configuration code below:

```yaml
llm:
  api_type: 'openai' # or azure / ollama / groq etc. Check LLMType for more options
  model: 'gpt-4-turbo' # or gpt-3.5-turbo
  base_url: 'https://api.openai.com/v1' # or forward url / other llm url
  api_key: 'YOUR_API_KEY'
```

## 3. Run Example Script
You can run the example script located at ML_Assistant/examples/di/machine_learning_with_tools.py and modify the requirements variable to conduct experiments.

