My idea is to generate our own text descriptors using Mordred, a Python library for the calculation of molecular descriptors, which are typically numeric values.

1.	We use Mordred to calculate a set of molecular descriptors for a given molecule. I’ve written code where I’ve picked 69 descriptors that I’ve used in the past on my SPECTRUM project. The ones included in my code would probably not be the optimal set but this is an illustration.
2.	I then apply the Mordred calculations to 100 randomly selected molecules from the firs 100,000 CIDs
3.	I then try to take each descriptor and find the top 5 molecules for that descriptor, for which I already have a text label. I describe that molecule as being “high” for that particular descriptor. This description gets appended to a growing total text description for that molecule.
4.	Ideally we would want to calculate Mordred descriptors for molecules selected using sampling statistics (10,000 molecules from say 1.5 million molecules?). We could determine “high” values based on the STDEV from the average for a given descriptor across each Mordred descriptor. 

Later on we could map these descriptors to textual descriptions using a machine learning-based approach. 

We could later train numerical values for each descriptor to a text label…using an ML approach. My domain knowledge in Chemistry for example might allow me to interpret the meaning of certain ranges of numerical descriptors so that labels could be generated. This would be time consuming to generate enough labels to train a model.

Obviously the quality and accuracy of the generated text descriptions will depend on the quality of the training data/labels and the complexity of the rules or models used.

What type of text descriptors would be valuable?
Biological Activity/Function for drug discovery and pharmaceutical research, potential therapeutic effects or interactions with biological targets.

Toxicity or Safety Information: potential risks or hazards, adverse effects, compatibility with specific applications.

Physical Properties: e.g. solubility, melting point, boiling point, and refractive index.

Structural Features: aromaticity, ring count, functional groups (leaving groups, applicability for a particular type of synthetic step), or substructures – these would be useful for different applications
