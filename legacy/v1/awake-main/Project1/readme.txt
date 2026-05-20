##### Models That Explain Themselves: Tree Algorithms #####

##### Project description #####
	  
	  This project seeks to provide insights on how machine learning models make their predictions. We explore how 
	  tree algorithms are built, tuned and then visualize the decision path taken on both correct and incorrect
	  decisions. 

	  The code was run on Ubuntu 22.04


##### Contents #####

      1. Installation

         The project assumes a working installation of python3. Steps are as follows:
	         - Open terminal and change working directory to the project folder.
	         - Create virtual environment 
	         	$ bash create_env
	         - Install Graphviz
	         	$ sudo apt-get install graphviz
	         - Create a kernel that can be used to run jupyter notebook commands inside the virtual environment
	         	$ python -m ipykernel install --user --name awaKE_1 --display-name "awaKE_1"
	         - Open jupyter notebook
	         - Open awaKE_1 and change the kernel to "awaKE_1"
	         - Run the notebook. Note that the trees will not be displayed unless all cells are ran.

	     To delete the installations run:
	         - $ deactivate
	         - $ jupyter kernelspec uninstall awaKE_1
	         - $ sudo apt-get purge --auto-remove graphviz
	         - $ rm -rf awaKE_1

	  2. Links & Useful Information

	     - Graphviz installation
	       https://graphviz.org/download/#linux
	       
	     - The dataset
	       https://github.com/plotly/datasets/blob/master/diabetes.csv
	       
	     - If you are using Anaconda you may encounter the error "EnvironmentLocationNotFound" when creating the environment.
	       The solution is to prevent Conda from activating the base environment by default as follows:
	         - $ conda config --set auto_activate_base false
	         - Start a new terminal
	         - Create the environment in step 1
	         - $ conda config --set auto_activate_base true   # To restore previous settings
	         
	     - Other sources
	       https://www.datacamp.com/tutorial/decision-tree-classification-python
	       https://github.com/parrt/dtreeviz/blob/master/notebooks/dtreeviz_sklearn_visualisations.ipynb
	       https://www.kdnuggets.com/2021/03/beautiful-decision-tree-visualizations-dtreeviz.html 
	       https://www.projectpro.io/recipes/optimize-hyper-parameters-of-decisiontree-model-using-grid-search-in-python
