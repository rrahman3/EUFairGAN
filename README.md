# Uncertainty Fairness Research Project

This repository contains the all the coding of the uncertainty fairness project. I tried to apply the opject-oriented programming (OOP) when coding. In a normal ml project, we just normally use the jupyter notebook, but for a big project like this, where we need to desgin multiple models and dataset and do different types of experiemtn and same code can be resued, a notebook is tough to scale. That's why I choose OOP principles.


## Project Structure
/ml_project 
    /data
    /notebooks
    /src 
    /data
        dataset.py # Custom Dataset class data_loader.py # Custom DataLoader class
    /models                  
        cnn_model.py         # CNN model
        resnet_model.py      # ResNet model
        vgg_model.py         # VGG model
        
    /preprocessing           
        preprocess.py        
        
    /training                
        trainer.py           # Trainer class
        
    /evaluation              
        evaluator.py         # Evaluate models
        
    /utils                   
        logger.py            # Logging utilities
        config_reader.py     # Configuration file reader
        
/configs                     
    datasets.yaml            # Configuration for datasets
    models.yaml              # Configuration for models and hyperparameters

/logs                        
/tests                       
main.py                      # Main function for running experiments
README.md                    
requirements.txt             # Required packages



## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ml_project

pip install -r requirements.txt

python main.py --model <model_name> --dataset <dataset_name>

python main.py --model cnn --dataset dataset1



### Explanation of the README Contents

1. **Project Structure**: 
   - Provides a clear overview of the project layout, which helps users navigate the code.

2. **Installation Instructions**: 
   - Steps to clone the repository and install dependencies, making it easy for users to set up the project.

3. **Configuration Section**: 
   - Explains how to configure datasets and models, guiding users in adding their custom datasets and models.

4. **Usage Instructions**: 
   - Clear command for running the main script with example usage for clarity.

5. **Adding New Models and Datasets**: 
   - Instructions for extending the project by adding new models and datasets, encouraging further development.

6. **Logging and Results**: 
   - Mentions logging capabilities and how users can modify them.

7. **Tests**: 
   - Encourages the addition of unit tests and specifies using a testing framework.

8. **License and Contribution Guidelines**: 
   - Provides legal information and invites contributions from others.

This `README.md` serves as a comprehensive guide for users and contributors, facilitating easier onboarding and usage of the project. Feel free to modify or extend it to fit your project's specifics.
