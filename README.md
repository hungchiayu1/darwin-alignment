# Reward Steering with Evolutionary Heuristics for Decoding-time Alignment
DARWIN is a decode-time alignment technique that uses a reward-guided tree search framework to align the LLM and achieve comparable performance to preference optimization on 2 instruction following benchmarks.

Paper Link: https://arxiv.org/abs/2406.15193


# How to use?
To run darwin, check out the demo notebook. You can run darwin with just a few lines of code!

To run evaluation on alpaca eval benchmark, you can use the following command
```
python3 alpaca_generate.py --method='darwin'  --model_name='meta-llama/Meta-Llama-3-8B-Instruct' --range='0-805' --replacement_period=40 --iteration=3 --n_mutation=1
```
The results will be saved in a json file where the 'past_outputs' contains a list of outputs for original output and iteration 1,2,3.

# Main results
<img width="638" alt="Screenshot 2024-07-08 at 5 31 18 PM" src="https://github.com/hungchiayu1/darwin-alignment/assets/72308196/6a39799a-13a8-4459-b3f0-46906433dd48">



<img width="381" alt="Screenshot 2024-07-08 at 5 35 30 PM" src="https://github.com/hungchiayu1/darwin-alignment/assets/72308196/ea584835-758c-4b09-aad5-5f566ae83caa">


# Overview of Darwin algorithm
<img width="538" alt="Screenshot 2024-07-08 at 5 33 26 PM" src="https://github.com/hungchiayu1/darwin-alignment/assets/72308196/204dd332-c1ad-45c3-b507-83e4e048719a">


# Citation
If you use Darwin in your publication, please cite it by using the following BibTeX entry.
```@misc{hung2024rewardsteeringevolutionaryheuristics,
      title={Reward Steering with Evolutionary Heuristics for Decoding-time Alignment}, 
      author={Chia-Yu Hung and Navonil Majumder and Ambuj Mehrish and Soujanya Poria},
      year={2024},
      eprint={2406.15193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15193}, 
}
```
