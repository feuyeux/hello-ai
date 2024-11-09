# Ray

<https://docs.ray.io/en/latest/index.html>

[Ray](https://github.com/ray-project/ray) is an open-source unified framework for scaling AI and Python applications. It provides the compute layer for parallel processing so that you don’t need to be a distributed systems expert.

## Ray framework

Ray’s unified compute framework consists of three layers:

1. **Ray AI Libraries**–An open-source, Python, domain-specific set of libraries that equip **ML engineers**, **data scientists**, and **researchers** with a scalable and unified toolkit for ML applications.
   1. [Data](https://docs.ray.io/en/latest/data/dataset.html): Scalable Datasets for ML
   2. [Train](https://docs.ray.io/en/latest/train/train.html): Distributed Training
   3. [Tune](https://docs.ray.io/en/latest/tune/index.html): Scalable Hyperparameter Tuning
   4. [RLlib](https://docs.ray.io/en/latest/rllib/index.html): Scalable Reinforcement Learning
   5. [Serve](https://docs.ray.io/en/latest/serve/index.html): Scalable and Programmable Serving

2. **Ray Core**–An open-source, Python, general purpose, distributed computing library that enables ML engineers and Python developers to scale Python applications and accelerate machine learning workloads.
   1. [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html): Stateless functions executed in the cluster.
   2. [Actors](https://docs.ray.io/en/latest/ray-core/actors.html): Stateful worker processes created in the cluster.
   3. [Objects](https://docs.ray.io/en/latest/ray-core/objects.html): Immutable values accessible across the cluster.

3. **Ray Clusters**–A set of worker nodes connected to a common Ray head node. Ray clusters can be fixed-size, or they can autoscale up and down according to the resources requested by applications running on the cluster.

![ray-air](https://docs.ray.io/en/latest/_images/ray-air.svg) **Ray AI Runtime(AIR)**

## Existing ML Platform integration

![air_arch_2](https://docs.ray.io/en/master/_images/air_arch_2.png)

In the above diagram:

1. A **workflow orchestrator** such as AirFlow, Oozie, SageMaker Pipelines, etc. is responsible for scheduling and creating Ray clusters and running Ray apps and services. The Ray application may be part of a larger orchestrated workflow (e.g., Spark ETL, then Training on Ray).

   Lightweight orchestration of task graphs can be handled entirely within Ray. External workflow orchestrators will integrate nicely but are only needed if running non-Ray steps.

2. Ray clusters can also **be created for interactive** use (e.g., Jupyter notebooks, Google Colab, Databricks Notebooks, etc.).

3. Ray Train, Data, and Serve provide integration with **Feature Stores** like Feast for Training and Serving.

4. Ray Train and Tune provide integration with **tracking** services such as MLFlow and Weights & Biases.
