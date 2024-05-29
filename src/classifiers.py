"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
import re
import torch

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

tmt = float(10)  # timeout

def search_result_rep(search_result: []):
    return [{**d["_source"],
             **{"id": d["_id"]},
             **{"score": d["_score"]}} for d in search_result]


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, t5_model, t5_tokenizer):
        self.device = "cuda"
        self.t5_model = t5_model.to(self.device)
        self.t5_tkz = t5_tokenizer
    
    
    def t5_validate(self, query: str, answer: str, score: float):
        text = query + " Document: " + answer + " Relevant: "
        input_ids = self.t5_tkz.encode(text,  return_tensors="pt").to(self.device)
        outputs=self.t5_model.generate(input_ids, eos_token_id=self.t5_tkz.eos_token_id, 
                                       max_length=64, early_stopping=True).to(self.device)
        outputs_decode = self.t5_tkz.decode(outputs[0][1:])
        outputs_logits=self.t5_model.generate(input_ids, output_scores=True, return_dict_in_generate=True, 
                                              eos_token_id=self.t5_tkz.eos_token_id, 
                                              max_length=64, early_stopping=True)
        sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
        t5_score = sigmoid_0[2].item()
        val_str = re.sub("</s>", "", outputs_decode)
        return {"Opinion": val_str, "Score": t5_score}
        