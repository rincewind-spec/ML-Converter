import coremltools as ct
import torch
from transformers import PegasusModel, PegasusConfig, PegasusTokenizer, PegasusForConditionalGeneration

# configuration = PegasusConfig()
# tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
# model = PegasusModel.from_pretrained("google/pegasus-large", torchscript=True)
# model.eval()
ARTICLE_TO_SUMMARIZE = "PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry " \
                        "conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " \
                        "scheduled to be affected by the shutoffs which were expected to last through at least " \
                        "midday tomorrow."
# traced_model = torch.jit.trace(model, ARTICLE_TO_SUMMARIZE)
# ml_model = ct.convert(traced_model, inputs=[tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='tf')])
# ml_model.save("TextSummarizer.mlmodel")

model_name = 'google/pegasus-large'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

model.eval()
traced_model = torch.jit.trace(model, example_inputs=torch.tensor([[0, 0], [0, 0]], dtype=torch.long))
ml_model = ct.convert(traced_model, inputs=[tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='tf')])
ml_model.save("TextSummarizer.mlmodel")

# batch = tokenizer.prepare_seq2seq_batch([ARTICLE_TO_SUMMARIZE], truncation=True, padding='longest', return_tensors="pt").to(torch_device)
# translated = model.generate(**batch)
# tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
# print(tgt_text)
