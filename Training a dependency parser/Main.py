from __future__ import unicode_literals, print_function

import plac
import random
import spacy
from pathlib import Path
from spacy.training.example import Example

TRAIN_DATA_CUSTOM = [
    ("She has many tasks to complete", {
        'heads': [1, 1, 3, 3, 3, 3],
        'deps': ['nsubj', 'ROOT', 'Quantity', 'dobj', 'mark', 'xcomp']
    }),
    ("The whole team celebrated their victory", {
        'heads': [1, 1, 3,3, 3, 3],
        'deps': ['det', 'amod', 'Quantity', 'nsubj', 'ROOT', 'dobj']
    }),
    ("There is enough time to finish the project", {
        'heads': [1, 1, 3, 3, 3, 3, 3, 3],
        'deps': ['expl', 'ROOT', 'Quantity', 'nsubj', 'aux', 'dobj', 'mark', 'xcomp']
    }),
    ("Numerous problems need to be", {
        'heads': [1, 1, 3, 3, 3],
        'deps': ['Quantity', 'nsubj', 'ROOT', 'amod', 'aux']
    }),
    ("Few opportunities are available", {
        'heads': [1, 1, 3, 3],
        'deps': ['Quantity', 'nsubj', 'ROOT', 'acomp']
    }),
    ("Many questions were asked", {
        'heads': [1, 1, 3, 3],
        'deps': ['Quantity', 'nsubjpass', 'ROOT', 'auxpass']
    }),
    ("Enough supplies were delivered", {
        'heads': [1, 1, 3, 3],
        'deps': ['Quantity', 'nsubjpass', 'ROOT', 'auxpass']
    }),
    ("The whole day", {
        'heads': [2, 2, 2],
        'deps': ['det', 'Quantity', 'amod']
    }),
    ("There is enough evidence to support claim", {
        'heads': [1, 1, 3, 3, 3, 3, 3],
        'deps': ['expl', 'ROOT', 'Quantity', 'nsubj', 'aux', 'dobj', 'mark']
    }),
    ("There are few apples the tree", {
        'heads': [1, 1, 5, 5, 5, 5],
        'deps': ['expl', 'ROOT', 'Quantity', 'dobj', 'prep', 'pobj']
    }),
    ("I need some rest", {
        'heads': [1, 1, 1, 3],
        'deps': ['nsubj', 'ROOT', 'Quantity', 'dobj']
    }),
    ("There are enough chairs for everyone .", {
        'heads': [3, 1, 3, 1, 5, 5, 5],
        'deps': ['punct', 'ROOT', 'Quantity', 'nsubj', 'prep', 'pobj', 'prep']
    }),
    ("Many people attended the event", {
        'heads': [1, 1, 3, 1, 1],
        'deps': ['Quantity', 'ROOT', 'nsubj', 'ROOT', 'dobj']
    }),
    ("All students passed the exam", {
        'heads': [1, 1, 3, 1, 3],
        'deps': ['Quantity', 'ROOT', 'nsubj', 'ROOT', 'dobj']
    }),
    ("you have some money", {
        'heads': [1, 1, 1, 3],
        'deps': ['nsubj', 'ROOT', 'Quantity', 'dobj']
    }),
    ("Half of the cake is gone", {
        'heads': [1, 1, 4, 4, 4, 4],
        'deps': ['Quantity', 'ROOT', 'det', 'nsubj', 'ROOT', 'acomp']
    }),
    ("Some of the books are missing", {
        'heads': [1, 1, 4, 4, 4, 4],
        'deps': ['Quantity', 'ROOT', 'prep', 'pobj', 'ROOT', 'acomp']
    }),
    ("All of the cookies are delicious .", {
        'heads': [1, 1, 5, 5, 5, 5, 5],
        'deps': ['Quantity', 'ROOT', 'prep', 'pobj', 'prep', 'acomp', 'punct']
    }),  
    ("Half of the job is done", {
        'heads': [1, 1, 4, 4, 4, 4],
        'deps': ['Quantity', 'ROOT', 'det', 'nsubj', 'ROOT', 'acomp']
    }),
    ("Some of the players are injured", {
        'heads': [1, 1, 4, 4, 4, 4],
        'deps': ['Quantity', 'ROOT', 'prep', 'pobj', 'ROOT', 'acomp']
    }),
    ("Few people understand the concept.", {
        'heads': [1, 1, 4, 4, 4, 4],
        'deps': ['Quantity', 'ROOT', 'det', 'nsubj', 'ROOT', 'dobj']
    }),
    ("All of the cars are parked.", {
        'heads': [1, 1, 5, 5, 5, 5, 5],
        'deps': ['Quantity', 'ROOT', 'prep', 'pobj', 'prep', 'acomp', 'punct']
    }),
    ("Many books are on the shelf", {
        'heads': [1, 1, 3, 1, 1, 1],
        'deps': ['Quantity', 'ROOT', 'nsubj', 'ROOT', 'prep', 'pobj']
    }),
]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int)
)
def main(model=None, output_dir=None, n_iter=70):
    """Load the model, set up the pipeline, and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  
        print("Created blank 'en' model")

    # Add the dependency parser component
    if 'parser' not in nlp.pipe_names:
        parser = nlp.add_pipe("parser")
    else:
        parser = nlp.get_pipe("parser")

    # add labels to the parser
    for _, annotations in TRAIN_DATA_CUSTOM:
        for dep in annotations.get('deps', []):
            parser.add_label(dep)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']

    with nlp.disable_pipes(*other_pipes):
        # train the parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA_CUSTOM)
            losses = {}
            for text, annotations in TRAIN_DATA_CUSTOM:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)
            print(losses)

    # test the trained model
    test_text = "you need some help"
    # test_text = "how many employees are working from home"
    # test_text = "All of you"
    doc = nlp(test_text)
    print('Dependencies', [(t.text, t.dep_, t.head.text) for t in doc])

    

if __name__ == '__main__':
    plac.call(main)


