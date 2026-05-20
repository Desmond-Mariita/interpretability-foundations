##### Models That Explain Themselves: Tree Algorithms #####

##### Project description #####
	  
	  This project seeks to provide insights on how deep learning models make their predictions.
	  We finetune a distilbert model on news classification data and discuss the explanations
	  usin LIME 
	  
	  The code was run on Ubuntu 22.04


##### Contents #####

      1. Installation

         The project assumes a working installation of python3. Steps are as follows:
	         - Open terminal and change working directory to the project folder.
	         - Create virtual environment 
	         	$ bash create_env
	         - Create a kernel that can be used to run jupyter notebook commands inside the virtual environment
	         	$ python -m ipykernel install --user --name awaKE_2 --display-name "awaKE_2"
	         - Open jupyter notebook
	         - Open awaKE_2 and change the kernel to "awaKE_2"
	   

	     To delete the installations run:
	         - $ deactivate
	         - $ jupyter kernelspec uninstall awaKE_2
	         - $ rm -rf awaKE_2

	  2. Links & Useful Information

	     - If you are using Anaconda you may encounter the error "EnvironmentLocationNotFound" when creating the environment.
	       The solution is to prevent Conda from activating the base environment by default as follows:
	         - $ conda config --set auto_activate_base false
	         - Start a new terminal
	         - Create the environment in step 1
	         - $ conda config --set auto_activate_base true   # To restore previous settings
	         
	
