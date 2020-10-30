# ERMI: Embedding-Rich Multiword Expression Identification

This repository contains the Verbal Multiword Expression (VMWE) Identification system submitted to the PARSEME shared task on semi-supervised identification of verbal multiword expressions - edition 1.2. 

Embedding-Rich Multiword Expression Identification (ERMI) is an embedding-rich **Bidirectional LSTM-CRF model**, which takes into account the embeddings of the word, its POS tag, dependency relation, and its head word.

## PARSEME shared task on semi-supervised identification of verbal multiword expressions - edition 1.2

PARSEME - edition 1.2 is a shared task on automatic verbal MWE (VMWE) identification, covering 14 languages (German (DE), Greek (EL), Basque (EU), French (FR), Irish (GA), Hebrew (HE), Hindi (HI), Italian (IT), Polish (PL), Brazilian Portuguese (PT), Romanian (RO), Swedish (SV), Turkish (TR), Chinese (ZH)). 

The [annotated corpora](http://hdl.handle.net/11234/1-3367) are provided in the .cupt format and include annotations of VMWEs in various categories (verbal idioms (VID), inherently reflexive verbs (IRV), light verb constructions with two subcategories (LVC.full and LVC.cause), verb-particle constructions with two subcategories (VPC.full and VPC.semi), inherently adpositional verbs (IAV), multi-verb constructions (MVC),  and inherently clitic verbs (LS.ICV)). 

In addition, [raw corpora](http://hdl.handle.net/11234/1-3416) without VMWE annotations for each language are provided. 

The focus of the third edition of this shared task is on discovering VMWEs that were not seen in the training corpus. 

## Embedding Models

Instead of using a pre-trained word embedding model, we've used the provided [raw corpora](http://hdl.handle.net/11234/1-3416) to train our own Fasttext word embedding models for each of the 14 languages. 

We provide the script that we use for creating the embedding models. 

       1. Please download the raw corpora from http://hdl.handle.net/11234/1-3416, or any other raw corpus in the conllu form will apply.
          Locate it into the Fasttext folder. 
       2. Indicate the language, in FasttextTrain.py, in the lines:
       
       language = "EL"
       and
       model_gensim.save("Fasttext/Embeddings/EL/gensim_el")
       
       3. Run FasttextTrain.py


## Bidirectional LSTM-CRF Model

ERMI is a system consisting of two supervised (ERMI, ERMI-head) and one semi-supervised (TeachERMI) neural network models, all of which carrying the same, bidirectional LSTM-CRF architecture. Each model consists of an input layer, LSTM layer and a CRF layer. 

For the input layer of ERMI, we use the concatenation of the embeddings of the word, its POS tag, its dependency relation to the head. For ERMI-head, we add to ERMI's input layer the embedding of the head of the word, where we aim to exploit the advantages the information of the head may provide us. TeachERMI is a teacher-student semi-supervised model, for which any of the previously mentioned input layers can be chosen. 

We've experimented with all three models for 14 languages on a validation set, and chosen the system with the highest VMWE identification performance for each language. Then, we submitted our official results for the annotated test corpora. Our overall system ranked 1st among 2 systems in the closed track, and 3rd among 9 systems in both
open and closed tracks with respect to the Unseen MWE-based F1 score. Our system also ranked 1st in the closed track for the HI, RO, TR and ZH languages in the Global MWE-based
F1 metric and 4th for all 14 languages among all systems in the Global MWE-based and Token-based F1 metric. ([Shared Task Results](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_02_MWE-LEX_2020___lb__COLING__rb__&subpage=CONF_50_Shared_task_results))

## Running ERMI

### Requirements
- Python 3.6
- Keras 2.2.4 with Tensorflow 1.12.0, and keras-contrib==2.0.8
- We cannot guarantee that the code works with different versions for Keras / Tensorflow.
- We cannot provide the data used in the experiments in this code repository, because we have no right to distribute the corpora provided by PARSEME Shared Task Edition 1.2 .

       1. Please download the annotated corpora from http://hdl.handle.net/11234/1-3367
          Unzip the downloaded file
          Locate it into input/corpora
       2. Locate the word embeddings that you've trained using the Fasttext script, and locate them into input/embeddings
       3. Language codes are German (DE), Greek (EL), Basque (EU), French (FR), Irish (GA), Hebrew (HE), Hindi (HI), Italian (IT), Polish (PL), Brazilian Portuguese (PT), Romanian (RO), Swedish (SV), Turkish (TR), Chinese (ZH).

#### Setup with virtual environment (Python 3):
-  python3 -m venv my_venv

   source my_venv/bin/activate
- Install the requirements:
   pip3 install -r requirements.txt

If everything works well, you can run the example usage described below.


### Example Usage:
- The following guide shows the example usages of the model.
- Instructions
      
      1. Change directory to the location of the source code
      2. Run the instructions in "Setup with virtual environment (Python 3)"
      3. Run the command to train the model: python3 src/Main.py -l GA -t gappy-crossy -e head -v 002 -w no
         -l is for the language code. Languages: BG, DE, EL, EN, ES, EU, FA, FR, HE, HI HR, HU, LT, IT, PL, PT, RO, SL, TR
         -t is for the tagging scheme. Tags: IOB, gappy-1, gappy-crossy
         -e is a flag for head-word embeddings. Flags: no, head
         -v is for version control. If you want to perform multiple trials, this will allow these trial to save the models in different folders.
         -w decides for the inclusion of the raw corpus. If -w yes, then semi-supervised learning through the teacher-student model is performed. If -w no, then no raw corpora is used during the training of the model. 
         
         
         For ERMI (without head-word embeddings), run the script with -e no -w no. Example: 
              python3 src/Main.py -l GA -t gappy-crossy -e no -v 001 -w no
         For ERMI-head (with head-word embeddings), run the script with -e head -w no. Example: 
              python3 src/Main.py -l GA -t gappy-crossy -e head -v 001 -w no
         For TeachERMI, run the script with -e no or -e head, according to your input layer preference. In addition, you should write -w yes, indicating that you will include the raw corpus in your training. First, run the teacher model. For example: 
              python3 src/Main.py -l GA -t gappy-crossy -e head -v 001 -w yes
         Afterwards, use the teacher model to tag/annotate a portion of the raw corpus. Then, concatenate the annotated raw corpus with the training dataset, and train the student model with this combined corpus. 
              python3 src/Main.py -l GA -t gappy-crossy -e head -v 002 -w yes
         
Any questions and issues are welcome, feel free to contact us at zeynep.yirmibesoglu@boun.edu.tr. 






