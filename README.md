# Project Text Analysis

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Authors](#authors)

## Introduction
In this project we analyze and compare human written articles and AI generated articles using NLP. In the analysis we cover syntax, semantics and pragmatics.

## Requirements
The following libraries are required for the files to run:
- Spacy
- Json
- Nlkt
- Fastcoref
- Spellchecker
- Asent
- Lingfeat
- spacytextblob

### Installing the requirements

**Clone the repository**
  ```
  git clone https://github.com/lmdeboer/PTA
  ```

  **Navigate to the PTA folder**
  ```
  cd PTA
  ```

  **Install the requirements**
  ```
  pip install -r requirements.txt
  ```

## Usage
After installing the requirements, run the script by using the following command:
 ```
python3 nlp.py 
 ```
## File Descriptions

### morphology.py
This file contains funtions for reading files, and performing basic preprocessing tasks like tokenization and lemmatization.

### syntax.py
This file contains functions related to syntax analyses, such as counting POS tags and noun chunks.

### semantics.py
This file contains functions to perform semantic analysis, such as counting unique synsets, ambiguous words and coreference resolution.

### pragmatics.py
This function cointains functions to perform pragmatic analysis such as sentiment analysis and discourse features.

### nlp.py
This is the main script that performs the NLP analysis by calling the functions from different modules, and writing the results to files for each part.

## Authors
Laura de Boer \
Dertje van Roggeveen\
Julian Paagman\
Roshana Vegter
