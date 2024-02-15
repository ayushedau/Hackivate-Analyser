
import spacy

nlp = spacy.load("en_core_web_lg")
import json
import os

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Concatenate the file name with the directory
file_path = os.path.join(current_dir, 'data.json')

with open(file_path, 'r') as f:
    data = json.load(f)

training_data = []
for example in data['examples']:
  temp_dict = {}
  temp_dict['text'] = example['content']
  temp_dict['entities'] = []
  for annotation in example['annotations']:
    start = annotation['start']
    end = annotation['end']
    label = annotation['tag_name'].upper()
    temp_dict['entities'].append((start, end, label))
  training_data.append(temp_dict)

from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en") # load a new spacy model
doc_bin = DocBin()

from spacy.util import filter_spans

for training_example  in tqdm(training_data): 
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents 
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy") 


nlp_ner = spacy.load("model-best")

doc = nlp_ner("The Chevrolet Camaro, a timeless classic with a legacy of speed and power, roars to life as its V8 engine ignites, sending shivers down the spine of anyone within earshot. Meanwhile, the Chevrolet Corvette, a true icon of American muscle, tears down the highway with unmatched ferocity, leaving all who witness it in awe of its sheer performance. But Chevrolet isn't just about raw power; it's also about practicality and versatility. The Chevrolet Silverado and Colorado trucks are renowned for their rugged durability and impressive towing capacity, making them the perfect companions for work and play alike. And let's not forget about the Chevrolet Equinox and Traverse SUVs, which offer comfort and convenience in equal measure, making family road trips a breeze. With a lineup that spans from compact cars to full-size trucks, Chevrolet truly has something for everyone, embodying the spirit of American ingenuity and innovation.")

colors = {"MAKENAME": "#F67DE3", "USERREVIEWS": "#7DF6D9", "MODELNAME": "#a6e22d"}
options = {"ents": ["MAKENAME", "USERREVIEWS", "MODELNAME"], "colors": colors}

spacy.displacy.render(doc, style="ent", options=options)

nlp = spacy.load("en_core_web_lg")
from spacy import displacy
html = displacy.render(doc, style="ent", options=options)
with open("data_vis.html", "w", encoding="utf-8") as f:
    f.write(html)