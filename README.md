# ESMattribution
Google colab notebooks to fine-tune ESM2 and obtain per residue attributions to that fine tuning.

A preprint is in progress and will be linked when available.

The underlying code is in the model_utils.py file.

train_attribute_fraction_alpha.ipynb contains code for training a model to predict the alpha helical fraction of a protein trained on the yeast proteome AlphaFold database predictions.  The training data is in yeast_af_ssstats2.csv.  The first cells of the notebook copy the code and training data files from the github.  Note that you have to have a uniprot id to map the attributions onto the AlphaFold database structure at the end of the notebook.

train_attribute_rna_binding.ipynb contains code for training a model to predict the whether a protein binds RNA.  The training data is in rna_binding_data.csv and is based on the RNApred dataset (https://webs.iiitd.edu.in/raghava/rnapred/download.html).

We also provide codes for calculating ESM2 representations without retraining and using ridge regression to calculate the contributions of each representation layer to the desired prediction.  ridge_attribute_fraction_alpha.ipynb illustrates that workflow for the alpha helix fraction described above.  ridge_attribute_rna_binding.ipynb shows the corresponding illustration for the RNA binding problem described above.
