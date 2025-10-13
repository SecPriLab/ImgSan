# ImgSan

This repository contains the source code and the **PrivacyVQA2K** dataset for the paper:

**_ImgSan: Utility-Based Image Privacy Sanitizer for Online Visual Question Answering_**

ImgSan is a tool designed to sanitize images by removing sensitive content while preserving essential information for Visual Question Answering (VQA) tasks.

## 1. Prerequisites

Before you begin, ensure you have the necessary models and environment set up.

1. **Install Segment Anything Model (SAM)**

   Install SAM to `ImgSan/ImgSan/models/` according to https://github.com/facebookresearch/segment-anything

2. **Install Grounding DINO**

   Install Grounding DINO to `ImgSan/ImgSan/models/` according to https://github.com/IDEA-Research/GroundingDINO

3. **Set Up Conda Environment**

   Once the models are in place, create the conda environment using the provided file:

   ```bash
   conda env create -f environment.yml
   ```

## 2. Code Execution

Follow these steps to run ImgSan and sanitize images.

1. **Activate Conda Environment**

   ```bash
   conda activate imgsan
   ```

2. **Run the Sanitization Script**

   ```bash
   python ImgSan/ImgSan/source/imgsan.py
   ```

3. **Find the Output**

   The sanitized images will be saved in the following directory:

   ```
   ImgSan/ImgSan/sanitized_images/
   ```

## 3. Privacy and Utility Evaluation

Scripts are provided to evaluate the privacy protection and utility preservation of the sanitized images.

### 3.1. Privacy Evaluation

This script evaluates the privacy metrics of the sanitized images.

1. **Run the Evaluation Script**

   ```bash
   python ImgSan/ImgSan/evaluation/evaluate_privacy.py
   ```

2. **Check the Results**

   The privacy evaluation results will be saved to:

   ```
   ImgSan/ImgSan/evaluation/privacy_results.json
   ```

### 3.2. Utility Evaluation

This script evaluates the VQA utility of the sanitized images using a specified model API.

1. **Configure API Credentials**

   Before running the script, open `ImgSan/ImgSan/evaluation/evaluate_utility.py` and configure your `api_key` and `base_url`.

2. **Run the Evaluation Script**

   ```bash
   python ImgSan/ImgSan/evaluation/evaluate_utility.py
   ```

3. **Check the Results**

   The utility evaluation results will be saved to:

   ```
   ImgSan/ImgSan/evaluation/utility_results.json
   ```