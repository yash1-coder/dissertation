[comment]: # (You may find the following markdown cheat sheet useful: https://www.markdownguide.org/cheat-sheet/. You may also consider using an online Markdown editor such as StackEdit or makeareadme.) 

## Project title: *Colouring Black Boxes - Visualization of Decision Making of AI*

### Student name: *Yash Chourasia*

### Student email: *yc432@student.le.ac.uk*

### Project description: 
*Deep learning models achieve strong performance in satellite image classification but often operate as black boxes, limiting interpretability and diagnostic insight. This project investigates how visualization-based explainability methods behave across three fundamentally different vision architectures: Convolutional Neural Networks (CNNs), Vision Transformers (ViT), and Vision Mamba models. Using the EuroSAT dataset containing labelled satellite images across ten land-use classes, one representative model from each architecture family will be trained and evaluated to establish baseline classification performance. Architecture-agnostic gradient-based explanation methods, including Integrated Gradients and SmoothGrad, will be implemented across all models. Architecture-specific approaches such as Grad-CAM for CNNs and attention-based visualisation for Transformers will be applied where appropriate. A structured evaluation framework will assess explanation quality using quantitative metrics for faithfulness, stability, and sensitivity under controlled perturbations, complemented by qualitative failure-case analysis. The primary contribution is a structured comparative analysis examining how architectural design influences explanation behaviour and evaluation metrics in remote sensing tasks.*

### List of requirements (objectives): 

[comment]: # (You can add as many additional bullet points as necessary by adding an additional hyphon symbol '-' at the end of each list) 

Essential:
- Conduct a focused literature review covering CNNs, Vision Transformers, State Space Models (Mamba), visualization-based XAI methods for computer vision, and explanation evaluation metrics.
- Prepare and preprocess the EuroSAT dataset with consistent train/validation/test splits suitable for all model architectures.
- Train and evaluate three representative model families on EuroSAT: a CNN (e.g., ResNet-18 or ResNet-50), a Vision Transformer (e.g., ViT-Tiny or ViT-Base), and a Vision Mamba model, documenting baseline classification performance.
- Implement architecture-agnostic visual explanation methods (Integrated Gradients and SmoothGrad) applicable across all models.
- Implement architecture-specific explanation methods (Grad-CAM / Grad-CAM++ for CNNs and attention-based visualisation for Vision Transformers where feasible).
- Develop and apply a structured evaluation framework incorporating quantitative metrics such as faithfulness (e.g., insertion/deletion tests), stability under input perturbations, and sensitivity to controlled modifications.
- Conduct systematic misclassification and failure-case analysis to identify shortcut learning, spurious correlations, and architecture-specific behavioural patterns.
- Produce a comparative analysis examining how convolutional, attention-based, and state-space modelling approaches influence explanation behaviour and evaluation results.

Desirable:
- Compare two variants within one architecture family (e.g., shallow vs deep ResNet or ViT-Tiny vs ViT-Base) to assess depth effects on explanation behaviour.
- Implement a limited model-agnostic explanation method (LIME or SHAP) for controlled cross-method comparison.
- Perform class-wise analysis to determine whether certain land-use categories exhibit differing explanation patterns across architectures.
- Develop an interactive Jupyter notebook to explore predictions, explanations, and failure cases.
- Extend quantitative evaluation with additional robustness or consistency metrics where feasible.

Optional:
- Conduct a small, informal human-centred interpretability study (subject to ethical approval) to assess perceived usefulness of explanations.
- Visualise internal feature representations using dimensionality reduction techniques (t-SNE or UMAP) to compare learned feature spaces.
- Investigate explanation robustness under controlled noise perturbations.
- Apply the evaluation framework to an additional satellite imagery dataset to assess generalisation.
- Analyse trade-offs between model accuracy, computational efficiency, and explanation quality.


## Information about this repository
This is the repository that you are going to use **individually** for developing your project. Please use the resources provided in the module to learn about **plagiarism** and how plagiarism awareness can foster your learning.

Regarding the use of this repository, once a feature (or part of it) is developed and **working** or parts of your system are integrated and **working**, define a commit and push it to the remote repository. You may find yourself making a commit after a productive hour of work (or even after 20 minutes!), for example. Choose commit message wisely and be concise.

Please choose the structure of the contents of this repository that suits the needs of your project but do indicate in this file where the main software artefacts are located.
