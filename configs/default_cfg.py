import spacy

from spacy.lang.tl import Tagalog



print("Loading Calamancy model for Tagalog...")
nlp = calamancy.load("tl_calamancy_md-0.1.0")
print("done")
# This dictionary structure directly mirrors the sections in your config.cfg file.
config = {
    "nlp": {
        "lang": "tl",
        "pipeline": ["tok2vec", "ner"],
        "batch_size": 1000,
        "tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"},
    },
    "components": {
        "tok2vec": {
            "factory": "tok2vec",
            "model": {
                "@architectures": "spacy.Tok2Vec.v2",
                "embed": {
                    "@architectures": "spacy.MultiHashEmbed.v2",
                    "width": 256, # This is inferred from 'encode.width'
                    "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                    "rows": [5000, 1000, 2500, 2500],
                    "include_static_vectors": True,
                },
                "encode": {
                    "@architectures": "spacy.MaxoutWindowEncoder.v2",
                    "width": 256,
                    "depth": 8,
                    "window_size": 1,
                    "maxout_pieces": 3,
                },
            },
        },
        "ner": {
            "factory": "ner",
            "scorer": {"@scorers": "spacy.ner_scorer.v1"},
            "model": {
                "@architectures": "spacy.TransitionBasedParser.v2",
                "state_type": "ner",
                "extra_state_tokens": False,
                "hidden_width": 64,
                "maxout_pieces": 2,
                "use_upper": True,
                "tok2vec": {
                    "@architectures": "spacy.Tok2VecListener.v1",
                    "width": 256, # This should match the tok2vec's width
                    "upstream": "*", # Use the pipeline's tok2vec component
                },
            },
        },
    },
    # The training block is not needed for model creation,
    # but would be used when calling nlp.evaluate or spacy.cli.train.
    # For a fully standalone script, you'd manage training separately.
}

# --- Create the NLP Pipeline ---

# Create a blank Tagalog pipeline
print("Creating blank 'tl' pipeline...")
nlp = spacy.blank("tl")

# Add the 'tok2vec' component from the config
print("Adding 'tok2vec' component...")
nlp.add_pipe("tok2vec", config=config["components"]["tok2vec"])

# Add the 'ner' component from the config
print("Adding 'ner' component...")
nlp.add_pipe("ner", config=config["components"]["ner"])

# --- Verification ---

# Print the pipeline component names to verify
print("\nPipeline created successfully!")
print("Pipeline components:", nlp.pipe_names)

# You can inspect the created components
ner_component = nlp.get_pipe("ner")
print("\nNER Model Architecture:")
print(ner_component.model)

# The 'nlp' object is now ready. You can proceed to load data,
# initialize its weights, and start the training loop.
# For example, to initialize the weights:
# nlp.initialize()
# print("\nPipeline initialized.")