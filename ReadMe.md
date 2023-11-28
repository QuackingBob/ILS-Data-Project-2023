# ILS Data Project

## The data

The raw data is from: 
- [ATOMATIC1111/Stable Diffusino WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Succinctly/Midjourney Prompts](https://huggingface.co/datasets/succinctly/midjourney-prompts)

All data can be found in the data directory. This directory has the following structure:

```bash
data/
│  
├──processed/
│  
└──raw/
   │  
   ├──artists.txt
   │  
   ├──flavors.txt
   │  
   ├──mediums.txt
   │  
   ├──movements.txt
   │  
   └──prompt data/
      │  
      ├──data/
      │  │  
      │  ├──test-00000-of-00001.parquet
      │  │  
      │  ├──train-00000-of-00001.parquet
      │  │  
      │  └──validation-00000-of-00001.parquet
      │  
      ├──dataset_infos.json
      │  
      ├──gitattributes
      │  
      └──README.md
```

Here is a breakdown of the files and directories:

- data (directory containint data)
    - processed (directory containing processed network graph data)
        - flavors_combine_redundant.json (file containing a json structure where each node has all the flavors that are very similar combined)
    - raw (directory containing raw prompt and dictionary data)
        - artists.txt (file containing a list of artists)
        - flavors.txt (file containing various prompt modifiers)
        - mediums.txt (file containing various artistic mediums)
        - movements.txt (file containing historical artistic movements)
        - prompt data (directory containing cloned midjourney prompt data)
            -dataset_infos.json (file with info about the data structure from original authors)
            -gitattributes (git log file from original repo)
            -README.md (readme about the prompt dataset from original author)
            -data (directory containing prompt data in train, test, val split, each as a parquet file)