import wandb
from pathlib import Path
import spacy

input_file = Path("entities/sample_text_dev.txt")

with open(input_file, 'r') as f:
    lines = f.readlines()
    
nlp = spacy.load("en_core_web_sm")

plots = []
for line in lines:
    print(line)
    doc = nlp(line)
    plots.append([wandb.plots.NER(docs=doc)])

with wandb.init(project="NeMo"):
    table = wandb.Table(data=plots, columns=["NER"])
    wandb.log({"NER Table": table})