# MLflow

**[MLflow](https://mlflow.org/)** is an open-source platform, purpose-built to assist **machine learning practitioners and teams** in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

MLflow is available in two main flavors: open source MLflow and [Managed MLflow](https://www.databricks.com/product/managed-mlflow), a service offered by Databricks, MLflow’s original creators. While both versions retain the core functionalities that MLflow is widely renowned for, they cater to different audiences and use cases.

![mlflow](tt/img/mlflow.png)

https://mlflow.org/docs/latest/introduction/index.html

## Core Components of MLflow

MLflow, at its core, provides a suite of tools aimed at simplifying the ML workflow. It is tailored to assist ML practitioners throughout the various stages of ML development and deployment. Despite its expansive offerings, MLflow’s functionalities are rooted in several foundational components:

### 1 [Tracking](https://mlflow.org/docs/latest/tracking.html#tracking)

MLflow Tracking provides both an API and UI dedicated to 

- the logging of parameters, 
- code versions, 
- metrics,
- and artifacts during the ML process. 

- This centralized repository captures details such as parameters, metrics, artifacts, data, and environment configurations, giving teams insight into their models’ evolution over time. 
- Whether working in standalone scripts, notebooks, or other environments, Tracking facilitates the logging of results either to local files or a server, making it easier to compare multiple runs across different users.

### 2 [Model Registry](https://mlflow.org/docs/latest/model-registry.html#registry)

A systematic approach to model management, the Model Registry assists in handling 

- different versions of models, 
- discerning their current state, 
- and ensuring smooth productionization. 

It offers a centralized model store, APIs, and UI to collaboratively manage an MLflow Model’s full lifecycle, including model lineage, versioning, aliasing, tagging, and annotations.

### 3 [MLflow Deployments for LLMs](https://mlflow.org/docs/latest/llms/deployments/index.html#deployments)

This server, equipped with a set of standardized APIs, streamlines access to both SaaS and OSS LLM models. It serves as a unified interface, bolstering security through authenticated access, and offers a common set of APIs for prominent LLMs.

### 4 [Evaluate](https://mlflow.org/docs/latest/models.html#model-evaluation)

Designed for in-depth model analysis, this set of tools facilitates objective model comparison, be it traditional ML algorithms or cutting-edge LLMs.

### 5 [Prompt Engineering UI](https://mlflow.org/docs/latest/llms/prompt-engineering/index.html#prompt-engineering)

A dedicated environment for prompt engineering, this UI-centric component provides a space for prompt experimentation, refinement, evaluation, testing, and deployment.

### 6 [Recipes](https://mlflow.org/docs/latest/recipes.html#recipes)

Serving as a guide for structuring ML projects, Recipes, while offering recommendations, are focused on ensuring functional end results optimized for real-world deployment scenarios.

### 7 [Projects](https://mlflow.org/docs/latest/projects.html#projects)

MLflow Projects standardize the packaging of ML code, workflows, and artifacts, akin to an executable. Each project, be it a directory with code or a Git repository, employs a descriptor or convention to define its dependencies and execution method.

[MLflow Deployments Server (Experimental) ‒ MLflow 2.10.2 documentation](https://mlflow.org/docs/latest/llms/deployments/index.html)

## Alternatives to open source MLflow

https://neptune.ai/blog/best-mlflow-alternatives

1. [neptune.ai](https://neptune.ai/)
2. [Azure Machine Learning](https://azure.microsoft.com/en-us/products/machine-learning)
3. [Weights & Biases](https://wandb.ai/site) (WandB) 
4. [Comet ML ](https://www.comet.com/)
5. [Valohai](https://valohai.com/)
6. [Metaflow](https://metaflow.org/)
7. [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
8. [Vertex AI](https://cloud.google.com/vertex-ai)