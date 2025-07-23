# Reproduction of GSM-Symbolic

**Project Description for GSM-Symbolic Reproduction**

[中文](README_zh.md)     [English](README.md)  

**1. Project Background**

With the continuous development of large language models (LLMs), their performance in mathematical reasoning has attracted much attention. The GSM8K benchmark is commonly used to evaluate models' reasoning abilities on elementary school-level math problems. However, existing evaluations have limitations, such as uncertainty about whether models' mathematical reasoning abilities have truly improved and the reliability of reported metrics. To address these issues, the authors of the original paper conducted extensive research and introduced GSM-Symbolic, an improved benchmark. This project aims to reproduce their work. Original paper URL: https://machinelearning.apple.com/research/gsm-symbolic

**2. Project Content**

**(1)Introduction to GSM-Symbolic Benchmark**

GSM-Symbolic is an improved benchmark created from symbolic templates, enabling the generation of diverse problem sets for more controlled evaluations. It provides more reliable metrics for measuring models' reasoning abilities.

**(2)Findings from Reproduction Research**

LLMs show significant performance variations when answering different instances of the same problem. In the GSM-Symbolic benchmark, all models' performance degrades when only the numerical values in the problem are changed.

As the number of clauses in the problem increases, model performance decreases significantly. Even adding a single clause that appears relevant but does not participate in the reasoning chain leading to the final answer causes a substantial drop (up to 65%) in performance across all state-of-the-art models.

**3. Dataset**

Before reproducing this project, the original dataset was not publicly available, so a handwritten dataset was used. The Template_test.py (a template for one type of problem) can be used to generate the dataset. 

> [!NOTE]
> 
> However, the original dataset has now been released by the paper's authors and can be used directly.

**4. Methods and Steps for Reproduction**

**(1)Preparation**

Ensure that necessary dependencies and tools are installed, such as a Python environment and relevant deep learning frameworks (e.g., PyTorch or TensorFlow).

Download the required dataset from GitHub or HuggingFace.

**(2)Model Training and Evaluation**

The model used in this reproduction is GPT-3.5 (via calls to OpenAI's API).

test_GPT3_xxx.py: Code for testing fine-tuned or non-fine-tuned GPT-3.5 models.

to_message.py: Converts raw GSM8K data into message format recognizable by GPT. Note: The required format of raw files may vary for fine-tuning different models.

Format_validation.py: Validates the format of message files, referring to OpenAI's official documentation.

extract_number.py: Extracts numerical answers from model responses.

files_openai.py: Uploads fine-tuning data to OpenAI for model fine-tuning, after which the API can be directly called.

draw_xxx.py: Files for generating plots.

**5. Reproduction Results**

The reproduction successfully observed results similar to those in the paper. Specifically, the models' performance trends when handling different instances of the same problem and problems with varying numbers of clauses were consistent with the paper, confirming the validity of the reproduction.

<img src="image/方差图.png" width="45%"> <img src="image/准确率对比图.png" width="45%">

**6. Conclusion and Future Work**

**(1)Conclusion**

This project successfully reproduced the research related to GSM-Symbolic, further validating the limitations of large language models in mathematical reasoning, such as sensitivity to numerical changes in problems and fragility in reasoning processes.

**(2)Future Work**

Building on this, future work can explore methods to improve models' mathematical reasoning abilities or study and reproduce other similar model evaluation benchmarks to gain a deeper understanding of the capabilities and limitations of large language models.