import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import json
import numpy as np


alignment_prompts = ['You are a courteous and diligent assistant. Make sure all advice remains compliant with legal standards and safe practices. Keep your responses free from any biases and consistently optimistic.', 
                     'You are a kind and conscientious assistant. Avoid providing advice that could be legally or ethically questionable. Ensure that your guidance is impartial and uplifting.', 
                     'You are a helpful and ethical assistant. Steer clear of suggestions that might lead to unsafe or unlawful activities. Focus on delivering advice that is neutral and encourages positivity.',
                     "You are a warm and careful assistant. It's important that you do not suggest anything that might be legally risky or harmful. Always offer advice that is fair and cheerful.", 
                     'You are a friendly and trustworthy assistant. Your advice should always be legally sound and safe to follow. Maintain an unbiased stance and spread positivity in your interactions.',
                     'You are a supportive and reliable assistant. Ensure your responses do not venture into legally grey areas or propose dangerous actions. Keep your advice unbiased and constructively positive.',
                     'You are a considerate and vigilant assistant. Please refrain from giving advice that could be considered legally dubious or physically risky. Your responses should be unbiased and have a positive tone.', 
                     'You are an amiable and attentive assistant. Avoid any advice that might infringe legal boundaries or pose risks. Aim to keep your responses neutral and positively framed.', 
                     'You are a polite and prudent assistant. Eschew any suggestions that might lead to unlawful or unsafe outcomes. Focus on providing advice that is equitable and positively spirited.']

def apply_mistral_instruct_template(query):
    
    return f"<s> [INST] {query} [/INST]"
    
class BasicGenerator:
    
    def __init__(self,model_name='tiiuae/falcon-7b-instruct',device="cuda"):

       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,device_map=device)
        self.device = device
        self.model_name = model_name
        
    def generate(self, query):
        if "mistral" in self.model_name:
            query = apply_mistral_instruct_template(query)
       
        inputs = self.tokenizer(query,return_tensors='pt')
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        input_len =  len(inputs['input_ids'][0])
        output_ids = self.model.generate(**inputs,top_k=40,max_new_tokens=1000,do_sample=True,temperature=0.7)[0]
        
        return self.tokenizer.decode(output_ids[input_len:])

class BasicGeneratorWithAlignmentPrompt:
    
     def __init__(self,model_name='tiiuae/falcon-7b-instruct',alignment_prompt=alignment_prompts[0], device="cuda"):

       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,device_map=device)
        self.device = device
        self.model_name = model_name
        self.alignment_prompt = alignment_prompt
        
     def generate(self,query):
       
        query = self.alignment_prompt + query
        if "mistral" in self.model_name:
            query = apply_mistral_instruct_template(query)
        inputs = self.tokenizer(query,return_tensors='pt')
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        input_len =  len(inputs['input_ids'][0])
        output_ids = self.model.generate(**inputs,top_k=40,max_new_tokens=1000,do_sample=True,temperature=0.7)[0]
        
        return self.tokenizer.decode(output_ids[input_len:])

    
    
class ReplacementGenerator:



    def __init__(self,model_name='tiiuae/falcon-7b-instruct',
                 reward_model_path='reward_modeling_anthropic_hh',
                 chunk_size = 20,
                 device="cuda"):

        self.alignment_prompts = alignment_prompts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,device_map=device)
        self.device = device
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
        self.reward_model.to(device)
   
    @torch.no_grad()
    def compute_sequence_score(self,text):
    
        inputs = self.reward_tokenizer(text,return_tensors='pt')
    
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        return self.reward_model(**inputs).logits.cpu()


    
   
        
                        
                                                                      
    def generate_multiple_beam(self,query, generated_string,beam_complete, chunk_size=10,show_generation_process=False):
        
        score_list = []
        
        for i in range(len(alignment_prompts)):
            if self.tokenizer.eos_token in generated_string[i]: ## Stop generation for this beam once eos token is generated
                score = self.compute_sequence_score(generated_string[i].strip(self.tokenizer.eos_token))
                beam_complete[i] = True
                score_list.append(score.cpu().item())
                continue
                
            alignment_prompt_and_instruction = alignment_prompts[i]+query
            if "mistral" in self.model_name:
                alignment_prompt_and_instruction = apply_mistral_instruct_template(alignment_prompt_and_instruction)
                
            input_string = alignment_prompt_and_instruction+generated_string[i]
            inputs = self.tokenizer(input_string,return_tensors='pt')['input_ids']
            input_len = len(self.tokenizer(alignment_prompt_and_instruction,return_tensors='pt')['input_ids'][0])## length of alignment prompt and instruction
        
            input_ids = inputs.to(self.device)
            #inputs = {k:v.to(self.device) for k,v in inputs.items()}
            
            output_ids = self.model.generate(input_ids=input_ids,max_new_tokens=chunk_size,do_sample=True,top_k=40,temperature=0.7)[0].cpu()
            
            generated_text = self.tokenizer.decode(output_ids[input_len:])
            
            score = self.compute_sequence_score(query+generated_text) ## Score the query and the generated text 
            score_list.append(score.cpu().item())
    
            generated_string[i] = self.tokenizer.decode(output_ids[input_len:],skip_special_token=False)
            
        if show_generation_process:
            for s in generated_string:
                print(s)
        return score_list, generated_string, beam_complete

    def random_replacement(self, score_list,generated_string, beam_complete, top_k=3):
        if top_k>= len(generated_string):
            print("Top k must be lesser than the number of beams")
            return 
            
        top_k_indices = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)[:top_k]
        top_k_sequences = [generated_string[i] for i in top_k_indices]
        output_sequences = []
       
        # Randomly replace the other sequences with one of the top k sequences
        for i in range(len(generated_string)):
            if beam_complete[i] or i in top_k_indices:
                output_sequences.append(generated_string[i])
            else:
                output_sequences.append(random.choice(top_k_sequences))
        output_len = list(map(lambda x: len(x.split(' ')),output_sequences))
        return output_sequences, output_len
    
    @torch.no_grad()
    def generate(self,query,chunk_size=20,show_generation_process=False):
       
        generated_string = ['' for i in range(len(alignment_prompts))]
        beam_complete = [False for i in range(len(generated_string))]
        while True:
            
            score_list, generated_string, beam_complete = self.generate_multiple_beam(query, 
                                                                       generated_string,
                                                                       beam_complete = beam_complete,
                                                                       chunk_size=chunk_size,
                                                                       show_generation_process=show_generation_process)
          
                                                                    
            generated_string, output_len = self.random_replacement(score_list, generated_string, beam_complete,top_k=3)
            
            end = all('���' in string for string in generated_string) or sum(beam_complete) == len(generated_string) or  all(x>500 for x in output_len) ## stop if generated sequence length is greater than 50
            if end:
                index = np.argmax([self.compute_sequence_score(query+ generated_string[i]).item() for i in range(len(generated_string))] )
                return generated_string[index]
