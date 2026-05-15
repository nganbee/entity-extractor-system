from datasets import load_dataset
import yaml
import os

def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# Truy cập vào label2id
label2id = config['labels']['label2id']

def mapping_data(ds):
    """Map dataset labels according to label2id"""
    id2label = {v: k for k, v in label2id.items()}
    
    def map_tags(example):
        # Convert ner_tags indices to label strings and back to ensure consistency
        mapped_tags = [id2label.get(tag, tag) for tag in example['ner_tags']]
        mapped_tags = [label2id.get(tag, tag) for tag in mapped_tags]
        example['ner_tags'] = mapped_tags
        return example
    
    ds = ds.map(map_tags, batched=False)
    return ds

def load_data():
    ds = load_dataset("Davlan/conll2003_noMISC")
    
    # Map the dataset according to config
    ds = mapping_data(ds)
    
    # Save to disk
    save_path = os.path.join(os.path.dirname(__file__), 'processed', 'conll2003_mapped')
    ds.save_to_disk(save_path)
    
    return ds

if __name__=="__main__":
    ds = load_data()
    
    print(f"Train length: {len(ds['train'])}")
    print(f"Validation length: {len(ds['validation'])}")
    print(f"Test length: {len(ds['test'])}")    