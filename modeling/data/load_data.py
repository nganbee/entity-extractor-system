from datasets import load_dataset
import os

def load_data():
    ds = load_dataset("Babelscape/multinerd", verification_mode="no_checks")

    ds_en = ds.filter(lambda example: example["lang"] == "en")
    
    # Save to disk
    save_path = os.path.join(os.path.dirname(__file__), 'processed', 'multinerd_en')
    ds_en.save_to_disk(save_path)
    
    return ds_en

if __name__=="__main__":
    ds = load_data()
    
    print(f"Train length: {len(ds['train'])}")
    print(f"Validation length: {len(ds['validation'])}")
    print(f"Test length: {len(ds['test'])}")    