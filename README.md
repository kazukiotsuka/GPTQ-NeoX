## GPTQ-NeoX
4 bits quantization for GPT-NeoX



## Installation
```bash
$ python3 -m venv venv
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install transformers==4.28.0
$ pip install torch==2.0.0
$ pip install torchaudio==2.0.1
$ pip install safetensors==0.3.0
```

## Usage

### Quantize
quantize to 4bit
```bash
CUDA_VISIBLE_DEVICES=0 python neox.py EleutherAI/gpt-neox-20b wikitext2 --wbits 4 --act-order --true-sequential --groupsize 128 --save gpt-neox-20b-4bit-128g.pt
```
 
### Benchmark
benchmark (fp16 baseline)
```bash
CUDA_VISIBLE_DEVICES=0 python neox.py EleutherAI/gpt-neox-20b wikitext2 --benchmark 2048 --check
```
 
benchmark (4bit)
```bash
CUDA_VISIBLE_DEVICES=0 python neox.py EleutherAI/gpt-neox-20b wikitext2 --wbits 4 --groupsize 128 --load gpt-neox-20b-4bit-128g.pt --benchmark 2048 --check
```

### Test Inference
test inference (fp16 baseline)
```bash
CUDA_VISIBLE_DEVICES=0 python neox_inference.py EleutherAI/gpt-neox-20b --text "The capital of Japan is"
```
 
test inference (4bit)
```bash

CUDA_VISIBLE_DEVICES=0 python neox_inference.py EleutherAI/gpt-neox-20b --wbits 4 --groupsize 128 --load gpt-neox-20b-4bit-128g.pt --text "The capital of Japan is"
```



## Acknowledgements
This code if a fork from [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) which is based on [GPTQ](https://github.com/IST-DASLab/gptq) by [IST-DASLab](https://github.com/IST-DASLab)   
[GPT-NeoX](https://huggingface.co/EleutherAI/gpt-neox-20b) is host by [EleutherAI](https://www.eleuther.ai/)  
Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)  
